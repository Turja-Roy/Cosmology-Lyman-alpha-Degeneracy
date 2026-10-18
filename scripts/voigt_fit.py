"""
Automatic Voigt decomposition, built to reproduce the collaborator's VPFIT convention.

The observed catalogue is a VPFIT fit; the simulated catalogue is a single-feature width
estimator. Those are different quantities, and no amount of noise matching makes them
comparable. This fits Voigt components the way VPFIT does so both sides can be measured
the same way.

The convention is readable off her own files. Every fitted region carries

    ! Stats:  25     1.4680027  105   96  0.002  0  AICc:     160.82

-- iterations, reduced chi-square, points, degrees of freedom, probability, AICc. Across
1624 regions the median reduced chi-square is 1.23. So: add components while AICc improves,
stop when it does not.

Because her spectra and her fits are both in data/Obs/, the fitter can be checked against
the thing it is imitating before it is pointed at a simulation:

    python scripts/voigt_fit.py --certify

fits her regions in her windows and reports how the recovered b-N catalogue compares to
hers, line by line. Nothing in data/Obs/ is written to.

    python scripts/voigt_fit.py --self-test
    python scripts/voigt_fit.py --certify --out output/obs/voigt_certify
    python scripts/voigt_fit.py --spectrum data/Obs/pg1048_all.dat --range 1200 1240
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import wofz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpfit_catalog import parse_all, normalise_ion  # noqa: E402

C_KMS = 299792.458

# Lyman-alpha. tau = TAU_PREFACTOR * N / b * H(a, u) with N in cm^-2 and b in km/s;
# a = DAMPING / b. Both follow from pi e^2 / (m_e c) * f * lambda0 / (sqrt(pi) * b).
LYA = 1215.67
TAU_PREFACTOR = 7.5804e-13
DAMPING = 6.061e-3

B_BOUNDS = (3.0, 200.0)
LOGN_BOUNDS = (11.0, 19.0)
MAX_COMPONENTS = 12

# A component sitting on a parameter bound is not a measurement: the fit has run out of
# room rather than converged, and the Jacobian column goes to zero so its error comes back
# as 0, which then sails through every error cut. The same failure put 31.5% of the
# simulated line widths at the top of the C++ b grid. Reject them instead.
RAIL_TOL_B = 2.0
RAIL_TOL_LOGN = 0.2


def voigt_tau(wave, z, b, logN):
    """Lyman-alpha optical depth of one component on `wave` [A]."""
    centre = LYA * (1.0 + z)
    u = (wave - centre) / centre * C_KMS / b
    a = DAMPING / b
    return TAU_PREFACTOR * 10.0 ** logN / b * wofz(u + 1j * a).real


def lsf_kernel(dv_pixel, fwhm_kms=17.0, table=None, table_dv=None):
    """Normalised LSF on the data's own pixel spacing.

    `table` is a tabulated COS kernel sampled every `table_dv` km/s; without one this is a
    Gaussian, which underfills the real wings and so under-predicts blending.
    """
    if table is not None:
        w = np.asarray(table, dtype=float)
        v_in = (np.arange(len(w)) - (len(w) - 1) / 2.0) * table_dv
        half = int(np.ceil(np.abs(v_in).max() / dv_pixel))
        k = np.interp(np.arange(-half, half + 1) * dv_pixel, v_in, w, left=0.0, right=0.0)
    else:
        sigma = fwhm_kms / 2.3548200 / dv_pixel
        half = max(int(np.ceil(4 * sigma)), 1)
        k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    total = k.sum()
    return k / total if total > 0 else np.array([1.0])


def convolve(flux, kernel):
    """Convolve holding the continuum outside the window, not wrapping around it.

    A fitting region is a window cut out of a longer spectrum, so the periodic padding used
    for simulated sightlines would fold one edge onto the other.
    """
    pad = len(kernel) // 2
    if pad == 0:
        return flux
    padded = np.concatenate([np.ones(pad), flux, np.ones(pad)])
    return np.convolve(padded, kernel, mode='same')[pad:-pad]


OVERSAMPLE = 5


def subpixel_grid(wave, oversample=OVERSAMPLE):
    """A uniform sub-grid with `oversample` points inside each data pixel.

    COS pixels here are ~8 km/s wide and the narrowest lines have b ~ 10 km/s, so a profile
    sampled at pixel centres is not the profile the detector integrated. Building the model
    on a sub-grid and averaging it down is what VPFIT does, and skipping it forces the fit
    to widen b to recover the absorption the centre sample missed.
    """
    dlam = (wave[-1] - wave[0]) / max(len(wave) - 1, 1)
    offsets = (np.arange(oversample) - (oversample - 1) / 2.0) * dlam / oversample
    return (wave[:, None] + offsets[None, :]).ravel()


def model_flux(wave_fine, params, kernel, n_pixels):
    """Absorbed, LSF-convolved flux averaged back down onto the data pixels."""
    tau = np.zeros_like(wave_fine)
    for i in range(0, len(params), 3):
        tau += voigt_tau(wave_fine, params[i], params[i + 1], params[i + 2])
    fine = convolve(np.exp(-np.clip(tau, 0.0, 500.0)), kernel)
    return fine.reshape(n_pixels, -1).mean(axis=1)


def _aicc(chi2, n, k):
    """AICc with the likelihood term already in chi-square, since the errors are known."""
    if n - k - 1 <= 0:
        return np.inf
    return chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _railed(params):
    """True if any component has run into a parameter bound."""
    for i in range(0, len(params), 3):
        b, logN = params[i + 1], params[i + 2]
        if b <= B_BOUNDS[0] + RAIL_TOL_B or b >= B_BOUNDS[1] - RAIL_TOL_B:
            return True
        if logN <= LOGN_BOUNDS[0] + RAIL_TOL_LOGN or logN >= LOGN_BOUNDS[1] - RAIL_TOL_LOGN:
            return True
    return False


def _bounds(wave, n_comp):
    zlo = wave[0] / LYA - 1.0
    zhi = wave[-1] / LYA - 1.0
    lo = np.tile([zlo, B_BOUNDS[0], LOGN_BOUNDS[0]], n_comp)
    hi = np.tile([zhi, B_BOUNDS[1], LOGN_BOUNDS[1]], n_comp)
    return lo, hi


def _refine(wave, flux, err, params, kernel, wave_fine):
    """Least-squares on all components at once, plus 1 sigma errors from the Jacobian."""
    lo, hi = _bounds(wave, len(params) // 3)
    start = np.clip(params, lo + 1e-12, hi - 1e-12)
    n_pixels = wave.size

    def residual(p):
        return (model_flux(wave_fine, p, kernel, n_pixels) - flux) / err

    fit = least_squares(residual, start, bounds=(lo, hi), method='trf', max_nfev=2000)
    chi2 = float(2.0 * fit.cost)

    # Covariance from the Gauss-Newton approximation; a singular Jacobian means the
    # component is unconstrained, which the error cuts downstream are there to catch.
    try:
        cov = np.linalg.pinv(fit.jac.T @ fit.jac)
        errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        errs = np.full_like(fit.x, np.inf)
    return fit.x, errs, chi2


# An isolated Lyman-alpha line at COS resolution has two near-equivalent solutions: narrow
# and saturated, or broad and optically thin. They differ by ~20 km/s in b and ~0.7 dex in
# N yet look alike once convolved, so whichever branch the starting guess sits on is the
# one the optimiser reports. VPFIT users break the tie with the higher Lyman series or by
# eye; an automatic fitter has to try both branches and let chi-square choose.
SEED_B = (10.0, 25.0, 60.0)


def _seeds(wave, flux, model, err):
    """Candidate starting points for the next component, one per curve-of-growth branch."""
    deficit = (model - flux) / err
    i = int(np.argmax(deficit))
    z = wave[i] / LYA - 1.0
    depth = np.clip(1.0 - flux[i], 1e-3, 0.999)
    out = []
    for b in SEED_B:
        # Invert the line centre, tau_0 = -ln(depth), at this branch's width.
        logN = np.log10(max(-np.log(1.0 - depth) * b / TAU_PREFACTOR,
                            10.0 ** LOGN_BOUNDS[0]))
        out.append([z, b, float(np.clip(logN, *LOGN_BOUNDS))])
    return out


def fit_region(wave, flux, err, fwhm_kms=17.0, kernel_table=None, kernel_dv=None,
               max_components=MAX_COMPONENTS):
    """Greedily add Voigt components to one window while AICc improves.

    Returns (params, errors, info). `params` is a flat (z, b, logN) vector.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    ok = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    wave, flux, err = wave[ok], flux[ok], err[ok]
    n = wave.size
    if n < 8:
        return np.array([]), np.array([]), {'n': n, 'chi2': np.nan, 'aicc': np.inf,
                                            'n_comp': 0}

    wave_fine = subpixel_grid(wave)
    dv_fine = np.median(np.diff(wave_fine)) / np.median(wave_fine) * C_KMS
    kernel = lsf_kernel(dv_fine, fwhm_kms, kernel_table, kernel_dv)

    params = np.array([])
    errs = np.array([])
    chi2 = float(np.sum(((1.0 - flux) / err) ** 2))
    best = _aicc(chi2, n, 0)
    info = {'n': n, 'chi2': chi2, 'aicc': best, 'n_comp': 0}

    model = np.ones_like(flux)
    for _ in range(max_components):
        p = e = None
        score = np.inf
        for seed in _seeds(wave, flux, model, err):
            try:
                tp, te, tc = _refine(wave, flux, err,
                                     np.concatenate([params, seed]), kernel, wave_fine)
            except (ValueError, np.linalg.LinAlgError):
                continue
            ts = _aicc(tc, n, len(tp))
            if ts < score and not _railed(tp):
                p, e, c, score = tp, te, tc, ts
        if p is None or not np.isfinite(score) or score >= best:
            break
        params, errs, chi2, best = p, e, c, score
        model = model_flux(wave_fine, params, kernel, n)
        info = {'n': n, 'chi2': chi2, 'aicc': best, 'n_comp': len(params) // 3}

    return params, errs, info


def components_from_fit(params, errs):
    """Flat parameter vector into per-component dicts, sorted by redshift."""
    out = []
    for i in range(0, len(params), 3):
        out.append({'z': params[i], 'z_err': errs[i],
                    'b': params[i + 1], 'b_err': errs[i + 1],
                    'logN': params[i + 2], 'logN_err': errs[i + 2]})
    return sorted(out, key=lambda c: c['z'])


# ----------------------------------------------------------------- certification


def hi_only_regions(all_path):
    """Her fitting windows that contain nothing but H I, keyed by (lo, hi)."""
    _, comps = parse_all(all_path)
    by_region = {}
    for c in comps:
        key = (c['region_lo'], c['region_hi'])
        by_region.setdefault(key, []).append(c)
    return {k: v for k, v in by_region.items()
            if np.isfinite(k[0]) and all(normalise_ion(c['ion']) == 'HI' for c in v)}


def spectrum_for(all_path):
    """The .dat her .all was fitted to: pg1048.all -> pg1048_all.dat."""
    stem = Path(all_path).stem
    path = Path(all_path).with_name(f'{stem}_all.dat')
    return path if path.exists() else None


def match_components(ours, theirs, dv_tol=15.0):
    """Pair components by velocity separation, nearest first, one-to-one."""
    pairs, used = [], set()
    for a in ours:
        best, best_dv = None, dv_tol
        for j, b in enumerate(theirs):
            if j in used:
                continue
            dv = abs(a['z'] - b['z']) / (1.0 + b['z']) * C_KMS
            if dv < best_dv:
                best, best_dv = j, dv
        if best is not None:
            used.add(best)
            pairs.append((a, theirs[best], best_dv))
    return pairs


def certify(paths, fwhm_kms=17.0, max_b_err=10.0, max_logN_err=0.5, out_dir=None):
    ours_all, theirs_all, pairs_all = [], [], []
    region_counts = []

    for all_path in paths:
        dat = spectrum_for(all_path)
        if dat is None:
            continue
        regions = hi_only_regions(all_path)
        if not regions:
            continue
        wave, flux, err = np.loadtxt(dat, unpack=True)
        order = np.argsort(wave)
        wave, flux, err = wave[order], flux[order], err[order]
        name = Path(all_path).stem

        for (lo, hi), theirs in regions.items():
            sel = (wave >= lo) & (wave <= hi)
            if sel.sum() < 8:
                continue
            params, errs, info = fit_region(wave[sel], flux[sel], err[sel], fwhm_kms)
            ours = components_from_fit(params, errs)
            for c in ours:
                c['sightline'] = name
            region_counts.append((len(ours), len(theirs)))
            ours_all += ours
            theirs_all += theirs
            pairs_all += match_components(ours, theirs)

    return _report(ours_all, theirs_all, pairs_all, region_counts,
                   max_b_err, max_logN_err, out_dir)


def _cut(rows, max_b_err, max_logN_err):
    return [r for r in rows
            if r['b_err'] <= max_b_err and r['logN_err'] <= max_logN_err]


def _summary(label, rows):
    b = np.array([r['b'] for r in rows])
    n = np.array([r['logN'] for r in rows])
    print(f'  {label:<12}{len(rows):>6}   b {np.median(b):6.1f} '
          f'[{np.percentile(b, 16):5.1f} {np.percentile(b, 84):5.1f}]   '
          f'logN {np.median(n):6.2f} [{np.percentile(n, 16):5.2f} {np.percentile(n, 84):5.2f}]')
    return b, n


def _report(ours, theirs, pairs, region_counts, max_b_err, max_logN_err, out_dir):
    from scipy.stats import ks_2samp

    ours_cut = _cut(ours, max_b_err, max_logN_err)
    theirs_cut = _cut(theirs, max_b_err, max_logN_err)
    if not ours_cut or not theirs_cut:
        print('nothing survived the error cuts')
        return 1

    print(f'\nH I-only regions fitted: {len(region_counts)}')
    print(f'components: {len(ours)} ours, {len(theirs)} VPFIT '
          f'(after b_err <= {max_b_err}, logN_err <= {max_logN_err}: '
          f'{len(ours_cut)} / {len(theirs_cut)})\n')

    print('  catalogue        n   b median [16-84]        logN median [16-84]')
    b_ours, n_ours = _summary('ours', ours_cut)
    b_theirs, n_theirs = _summary('VPFIT', theirs_cut)

    kb = ks_2samp(b_ours, b_theirs)
    kn = ks_2samp(n_ours, n_theirs)
    print(f'\n  b    KS D = {kb.statistic:.3f}, p = {kb.pvalue:.3g}'
          f'   median ratio {np.median(b_ours) / np.median(b_theirs):.3f}')
    print(f'  logN KS D = {kn.statistic:.3f}, p = {kn.pvalue:.3g}'
          f'   median offset {np.median(n_ours) - np.median(n_theirs):+.3f} dex')

    delta = np.array([a - b for a, b in region_counts])
    print(f'\n  components per region: ours - VPFIT, mean {delta.mean():+.2f}, '
          f'median {np.median(delta):+.0f}, '
          f'same in {100 * np.mean(delta == 0):.0f}% of regions')

    if pairs:
        db = np.array([a['b'] - b['b'] for a, b, _ in pairs])
        dn = np.array([a['logN'] - b['logN'] for a, b, _ in pairs])
        print(f'\n  matched within 15 km/s: {len(pairs)} of {len(theirs)} VPFIT components '
              f'({100 * len(pairs) / len(theirs):.0f}%)')
        print(f'    b residual    median {np.median(db):+6.2f} km/s, scatter '
              f'{0.5 * (np.percentile(db, 84) - np.percentile(db, 16)):.2f}')
        print(f'    logN residual median {np.median(dn):+6.3f} dex,  scatter '
              f'{0.5 * (np.percentile(dn, 84) - np.percentile(dn, 16)):.3f}')

    verdict = (abs(np.median(b_ours) / np.median(b_theirs) - 1) < 0.10
               and kb.pvalue > 0.01)
    print(f'\n  CERTIFIED: {verdict}  '
          f'(b median within 10% and KS not rejected at 1%)')

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(ours, out / 'voigt_auto_components.csv')
        _plot(b_ours, b_theirs, n_ours, n_theirs, pairs, out / 'certification.png')
        print(f'  written: {out}/voigt_auto_components.csv, {out}/certification.png')
    return 0


def _write_csv(rows, path):
    import csv
    fields = ['sightline', 'z', 'z_err', 'b', 'b_err', 'logN', 'logN_err']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _plot(b_ours, b_theirs, n_ours, n_theirs, pairs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (a, c, label) in zip(axes[:2],
                                 [(b_ours, b_theirs, r'$b$ [km s$^{-1}$]'),
                                  (n_ours, n_theirs, r'$\log N_{\rm HI}$')]):
        for data, name, colour in [(c, 'VPFIT', 'k'), (a, 'automatic', 'C3')]:
            x = np.sort(data)
            ax.step(x, np.arange(1, x.size + 1) / x.size, where='post',
                    color=colour, lw=1.6, label=name)
        ax.set_xlabel(label)
        ax.set_ylabel('cumulative fraction')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    ax = axes[2]
    if pairs:
        ax.scatter([t['b'] for _, t, _ in pairs], [o['b'] for o, _, _ in pairs],
                   s=10, alpha=0.5, color='C0')
        lim = [0, max(120, max(t['b'] for _, t, _ in pairs))]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel(r'VPFIT $b$ [km s$^{-1}$]')
    ax.set_ylabel(r'automatic $b$ [km s$^{-1}$]')
    ax.set_title('matched components')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- self-test


def self_test():
    rng = np.random.default_rng(7)
    wave = np.arange(1220.0, 1226.0, 0.00997)      # COS G130M sampling

    truth = [(0.0045, 22.0, 13.9), (0.0068, 35.0, 14.3)]
    fine = subpixel_grid(wave)
    tau = sum(voigt_tau(wave, *t) for t in truth)
    tau_fine = sum(voigt_tau(fine, *t) for t in truth)
    dv = np.median(np.diff(fine)) / np.median(fine) * C_KMS
    kernel = lsf_kernel(dv, 17.0)
    clean = convolve(np.exp(-tau_fine), kernel).reshape(wave.size, -1).mean(axis=1)

    # tau_0 = 0.758 (N/1e13)(10/b) is the textbook Lyman-alpha line centre.
    tau0 = voigt_tau(np.array([LYA * 1.0001]), 0.0001, 10.0, 13.0)[0]
    assert abs(tau0 - 0.758) < 0.01, tau0

    # The LSF conserves equivalent width while making lines shallower.
    ew_in = np.trapezoid(1.0 - np.exp(-tau), wave)
    ew_out = np.trapezoid(1.0 - clean, wave)
    assert abs(ew_out / ew_in - 1.0) < 0.01, (ew_in, ew_out)
    assert clean.min() > np.exp(-tau).min()

    sigma = np.full_like(wave, 0.02)
    flux = clean + rng.normal(0.0, sigma)
    params, errs, info = fit_region(wave, flux, sigma)
    comps = components_from_fit(params, errs)
    assert info['n_comp'] == 2, info
    assert 0.8 < info['chi2'] / (info['n'] - 6) < 1.3, info

    for got, want in zip(comps, truth):
        assert abs(got['z'] - want[0]) / (1 + want[0]) * C_KMS < 5.0, (got, want)
        assert abs(got['b'] - want[1]) < 5.0, (got, want)
        assert abs(got['logN'] - want[2]) < 0.15, (got, want)

    # A pure-noise window must not invent components: AICc has to reject the first one.
    empty, _, einfo = fit_region(wave, 1.0 + rng.normal(0.0, sigma), sigma)
    assert einfo['n_comp'] == 0, einfo

    # Noise alone decides what survives: the same weak line is found at one noise level
    # and lost at another. This is the detection bias the whole comparison hinges on.
    weak = convolve(np.exp(-voigt_tau(fine, 0.005, 25.0, 12.6)),
                    kernel).reshape(wave.size, -1).mean(axis=1)
    for sigma_w, expected in [(0.05, 1), (0.20, 0)]:
        s_w = np.full_like(wave, sigma_w)
        _, _, winfo = fit_region(wave, weak + rng.normal(0.0, sigma_w, wave.size), s_w)
        assert winfo['n_comp'] == expected, (sigma_w, winfo)

    # A component pinned to a bound is rejected rather than reported with a zero error.
    assert _railed(np.array([0.005, B_BOUNDS[1] - 1.0, 13.0]))
    assert _railed(np.array([0.005, B_BOUNDS[0] + 1.0, 13.0]))
    assert not _railed(np.array([0.005, 25.0, 13.0]))

    assert match_components([{'z': 0.0045}], [{'z': 0.00451}])
    assert not match_components([{'z': 0.0045}], [{'z': 0.0070}])

    print('voigt_fit self-test passed')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--certify', action='store_true',
                   help="fit the collaborator's own windows and compare to her catalogue")
    p.add_argument('--fits', default='data/Obs/*.all')
    p.add_argument('--spectrum', help='fit one .dat file over --range')
    p.add_argument('--range', nargs=2, type=float, metavar=('LO', 'HI'))
    p.add_argument('--fwhm', type=float, default=17.0, help='Gaussian LSF FWHM [km/s]')
    p.add_argument('--max-b-err', type=float, default=10.0)
    p.add_argument('--max-logN-err', type=float, default=0.5)
    p.add_argument('--out', default='output/obs/voigt_certify')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.spectrum:
        wave, flux, err = np.loadtxt(args.spectrum, unpack=True)
        if args.range:
            sel = (wave >= args.range[0]) & (wave <= args.range[1])
            wave, flux, err = wave[sel], flux[sel], err[sel]
        params, errs, info = fit_region(wave, flux, err, args.fwhm)
        print(f"{info['n_comp']} components, chi2/dof = "
              f"{info['chi2'] / max(info['n'] - len(params), 1):.3f}")
        for c in components_from_fit(params, errs):
            print(f"  z {c['z']:.6f} +- {c['z_err']:.6f}   "
                  f"b {c['b']:6.2f} +- {c['b_err']:5.2f}   "
                  f"logN {c['logN']:6.3f} +- {c['logN_err']:.3f}")
        return 0

    if args.certify:
        paths = sorted(glob.glob(args.fits))
        if not paths:
            p.error(f'no fits matched {args.fits}')
        return certify(paths, args.fwhm, args.max_b_err, args.max_logN_err, args.out)

    p.error('need --certify, --spectrum or --self-test')


if __name__ == '__main__':
    sys.exit(main())
