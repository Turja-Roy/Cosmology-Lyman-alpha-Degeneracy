# ============================================================ #
# DEPRECATED LEGACY FILE - DO NOT IMPORT
# ============================================================ #
# This file contains old utility functions that have been
# migrated to other modules. It is kept for reference only.
# 
# Active functions have been moved to:
# - scripts/fake_spectra_fix.py (compute_temp_density_chunked)
# - scripts/commands/analyze.py (VoigtFit functions)
# - scripts/analysis.py (analysis functions)
# - scripts/plotting.py (plotting functions)
#
# Last updated: 2026-01-29
# ============================================================ #

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


# HDF5 UTILITIES

def load_snapshot_metadata(filepath):
    with h5py.File(filepath, 'r') as f:
        header = f['Header']

        metadata = {
            'redshift': float(header.attrs['Redshift']),
            'boxsize': float(header.attrs['BoxSize']),  # ckpc/h
            'hubble': float(header.attrs['HubbleParam']),
            'omega_matter': float(header.attrs['Omega0']),
            'omega_lambda': float(header.attrs['OmegaLambda']),
            'num_gas': int(header.attrs['NumPart_ThisFile'][0]),
            'num_dm': int(header.attrs['NumPart_ThisFile'][1]),
            'num_stars': int(header.attrs['NumPart_ThisFile'][4]),
            'time': float(header.attrs['Time']),
        }

        # Derived quantities
        metadata['boxsize_mpc'] = metadata['boxsize'] / \
            metadata['hubble'] / 1000  # Mpc/h comoving
        metadata['boxsize_proper'] = metadata['boxsize'] / \
            metadata['hubble'] / (1 + metadata['redshift']
                                  ) / 1000  # Mpc proper

    return metadata

def load_gas_properties(filepath, fields=None, stride=1, max_particles=None):
    with h5py.File(filepath, 'r') as f:
        if 'PartType0' not in f:
            raise ValueError("No gas particles (PartType0) in snapshot")

        gas = f['PartType0']

        # If fields is None, include all
        if fields is None:
            fields = list(gas.keys())

        # Determine slice
        if max_particles is not None:
            end = min(max_particles * stride, len(gas['Coordinates']))
            slice_obj = slice(0, end, stride)
        else:
            slice_obj = slice(None, None, stride)

        # Load data
        data = {}
        for field in fields:
            if field in gas:
                data[field] = gas[field][slice_obj]
            else:
                print(f"Warning: Field '{field}' not found in snapshot")

        # Add metadata
        data['n_particles'] = len(data[list(data.keys())[0]])

    return data

def explore_hdf5_structure(filepath):
    structure = {'groups': [], 'header': {}}

    with h5py.File(filepath, 'r') as f:
        # Top-level groups
        structure['groups'] = list(f.keys())

        print(f"\n{'='*70}")
        print(f"HDF5 Structure: {os.path.basename(filepath)}")
        print(f"{'='*70}")
        print(f"\nTop-level groups: {structure['groups']}")

        # Header info
        if 'Header' in f:
            header = f['Header']
            print(f"\nHeader attributes:")
            for key in header.attrs.keys():
                structure['header'][key] = header.attrs[key]

        # Particle type info
        for ptype in ['PartType0', 'PartType1', 'PartType4', 'PartType5']:
            if ptype in f:
                datasets = list(f[ptype].keys())
                structure[ptype] = datasets

                ptype_name = {
                    'PartType0': 'GAS',
                    'PartType1': 'DARK MATTER',
                    'PartType4': 'STARS',
                    'PartType5': 'BLACK HOLES'
                }.get(ptype, ptype)

                print(f"\n{ptype} ({ptype_name}):")
                for ds in datasets:
                    shape = f[ptype][ds].shape
                    dtype = f[ptype][ds].dtype
                    print(f"  {ds:30s} shape={str(shape):20s} dtype={dtype}")

    return structure


# ===================================== #
# FAKE_SPECTRA BUGFIXES FOR PYTHON 3.13 #
# ===================================== #

def apply_fake_spectra_bugfixes():
    # Fixes for:
    # 1. uint32 overflow in get_npart calculation
    # 2. float32/float64 type mismatches in C extension
    try:
        from fake_spectra import abstractsnapshot
        from fake_spectra import spectra
        from fake_spectra._spectra_priv import _Particle_Interpolate as _PI_original

        # FIX 1: uint32 overflow
        def get_npart_fixed(self):
            """Get the total number of particles (fixed for uint32 overflow)."""
            npart_total = self.get_header_attr(
                "NumPart_Total").astype(np.int64)
            npart_high = self.get_header_attr(
                "NumPart_Total_HighWord").astype(np.int64)
            return npart_total + (2**32) * npart_high

        abstractsnapshot.AbstractSnapshotFactory.get_npart = get_npart_fixed
        abstractsnapshot.HDF5Snapshot.get_npart = get_npart_fixed
        abstractsnapshot.BigFileSnapshot.get_npart = get_npart_fixed

        # FIX 2: float32/float64 type casting
        def _do_interpolation_work_fixed(self, pos, vel, elem_den, temp, hh, amumass, line, get_tau):
            """Run the interpolation with proper float32 casting (fixed for Python 3.13)"""
            if self.turn_off_selfshield:
                gamma_X = 0
            else:
                gamma_X = line.gamma_X

            # Ensure all scalar parameters are float32
            box = np.float32(self.box)
            velfac = np.float32(self.velfac)
            atime = np.float32(self.atime)
            lambda_X = np.float32(line.lambda_X * 1e-8)
            gamma_X_f32 = np.float32(gamma_X)
            fosc_X = np.float32(line.fosc_X)
            amumass_f32 = np.float32(amumass)
            tautail = np.float32(self.tautail)

            # Ensure all array parameters are float32 (except cofm which needs float64)
            pos = np.asarray(pos, dtype=np.float32)
            vel = np.asarray(vel, dtype=np.float32)
            elem_den = np.asarray(elem_den, dtype=np.float32)
            temp = np.asarray(temp, dtype=np.float32)
            hh = np.asarray(hh, dtype=np.float32)
            axis = np.asarray(self.axis, dtype=np.int32)
            # cofm must be float64!
            cofm = np.asarray(self.cofm, dtype=np.float64)

            return _PI_original(get_tau*1, self.nbins, self.kernel_int, box, velfac, atime,
                                lambda_X, gamma_X_f32, fosc_X, amumass_f32, tautail,
                                pos, vel, elem_den, temp, hh, axis, cofm)

        spectra.Spectra._do_interpolation_work = _do_interpolation_work_fixed

        print("Applied fake_spectra bugfixes for Python 3.13 compatibility")
        return True

    except ImportError:
        print("Warning: fake_spectra not installed - skipping bugfixes")
        return False


# ===================#
# PLOTTING UTILITIES #
# ===================#

def setup_plot_style():
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


def save_plot(fig, filepath, dpi=150):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {filepath}")


def create_sample_spectra_plot(velocity, flux, redshift, n_samples=5, output_path=None):
    n_samples = min(n_samples, flux.shape[0])
    indices = np.random.choice(flux.shape[0], n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 2*n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        ax.plot(velocity, flux[idx], 'k-', lw=0.8, alpha=0.8)
        ax.set_ylabel('Flux', fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1.0, color='r', ls='--', alpha=0.3, lw=1)
        ax.axhline(0.0, color='gray', ls='--', alpha=0.3, lw=1)
        ax.grid(alpha=0.2)
        ax.set_title(f'Sightline {idx} (z={redshift:.2f})', fontsize=11)

        if i == n_samples - 1:
            ax.set_xlabel('Velocity [km/s]', fontsize=11)

    plt.suptitle(f'CAMEL Lyman-α Spectra (z={redshift:.2f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        save_plot(fig, output_path)

    return fig, axes


# ================#
# DATA PROCESSING #
# ================#

# Compute flux statistics from optical depth.
def compute_flux_statistics(tau):
    flux = np.exp(-tau)

    stats = {
        'mean_flux': float(np.mean(flux)),
        'median_flux': float(np.median(flux)),
        'std_flux': float(np.std(flux)),
        'min_flux': float(np.min(flux)),
        'max_flux': float(np.max(flux)),
        'mean_tau': float(np.mean(tau)),
        'median_tau': float(np.median(tau)),
        'effective_tau': float(-np.log(np.mean(flux))),
    }

    # Absorption statistics
    total_pixels = flux.size
    stats['deep_absorption_frac'] = float((flux < 0.1).sum() / total_pixels)
    stats['moderate_absorption_frac'] = float(
        ((flux >= 0.1) & (flux < 0.5)).sum() / total_pixels)
    stats['weak_absorption_frac'] = float((flux >= 0.5).sum() / total_pixels)

    return stats


# Compute 1D flux power spectrum P_F(k).
def compute_power_spectrum(flux, velocity_spacing, chunk_size=1000):
    """
    Compute 1D flux power spectrum P_F(k).
    
    Parameters
    ----------
    flux : ndarray, shape (n_sightlines, n_pixels)
        Flux array
    velocity_spacing : float
        Velocity spacing per pixel (km/s)
    chunk_size : int, optional
        Number of sightlines to process at once (default: 1000)
        Reduces memory usage for large datasets
    
    Returns
    -------
    dict with keys:
        k : wavenumber array (s/km)
        P_k_mean : mean power spectrum
        P_k_std : standard deviation
        P_k_err : standard error
        mean_flux : mean flux
        n_modes : number of independent modes
        n_sightlines : number of sightlines
        velocity_spacing : velocity spacing
    """
    n_sightlines, n_pixels = flux.shape

    # Normalize flux to get flux contrast
    mean_flux = np.mean(flux)
    
    # Wavenumber array (s/km units)
    k = np.fft.rfftfreq(n_pixels, d=velocity_spacing)
    n_k = len(k)
    
    # Initialize accumulators for mean and variance computation
    # Using Welford's online algorithm to avoid storing all power spectra
    power_sum = np.zeros(n_k)
    power_sum_sq = np.zeros(n_k)
    
    # Process in chunks to reduce memory usage
    n_chunks = int(np.ceil(n_sightlines / chunk_size))
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_sightlines)
        
        # Get flux chunk and compute contrast
        flux_chunk = flux[start_idx:end_idx]
        delta_F_chunk = flux_chunk / mean_flux - 1.0
        
        # Compute power for this chunk
        for i in range(delta_F_chunk.shape[0]):
            # FFT of flux contrast
            flux_fft = np.fft.rfft(delta_F_chunk[i])
            # Power spectrum (dimensionless, normalized by pixel count)
            power = np.abs(flux_fft)**2 / n_pixels
            
            # Update running sums
            power_sum += power
            power_sum_sq += power**2

    # Compute mean and standard deviation
    P_k_mean = (power_sum / n_sightlines) * velocity_spacing
    
    # Variance: Var(X) = E[X^2] - E[X]^2
    mean_power = power_sum / n_sightlines
    mean_power_sq = power_sum_sq / n_sightlines
    variance = mean_power_sq - mean_power**2
    P_k_std = np.sqrt(np.maximum(variance, 0)) * velocity_spacing  # Ensure non-negative
    P_k_err = P_k_std / np.sqrt(n_sightlines)
    
    # Number of independent modes (useful for error estimation)
    n_modes = np.ones_like(k) * n_sightlines

    return {
        'k': k,
        'P_k_mean': P_k_mean,
        'P_k_std': P_k_std,
        'P_k_err': P_k_err,
        'mean_flux': mean_flux,
        'n_modes': n_modes,
        'n_sightlines': n_sightlines,
        'velocity_spacing': velocity_spacing
    }


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_snapshot_number(filepath):
    basename = os.path.basename(filepath)
    if 'snap_' in basename:
        num_str = basename.split('snap_')[1].split('.')[0]
        return int(num_str)
    return None


# Compute column density distribution f(N_HI) from optical depth.
def compute_column_density_distribution(tau, velocity_spacing, threshold=0.5, colden=None):
    """See scripts/analysis.py for full documentation."""
    c = 2.998e5  # km/s
    lambda_lya = 1215.67  # Angstroms
    f_osc = 0.4162  # Oscillator strength for Lyman-alpha

    # Corrected constant (see scripts/analysis.py for details)
    TAU_TO_COLDEN_CONSTANT = 8.51e11  # cm^-2 / (km/s)

    column_densities = []

    for i in range(tau.shape[0]):
        tau_line = tau[i, :]
        colden_line = colden[i, :] if colden is not None else None

        # Find absorption features (contiguous pixels above threshold)
        absorbing = tau_line > threshold

        # Label connected regions
        in_feature = False
        feature_start = 0

        for j in range(len(tau_line)):
            if absorbing[j] and not in_feature:
                # Start of new feature
                in_feature = True
                feature_start = j
            elif not absorbing[j] and in_feature:
                # End of feature
                in_feature = False
                
                # Compute column density (use peak, not sum)
                # The colden array contains column density per pixel in cm^-2
                if colden_line is not None:
                    N_HI = np.max(colden_line[feature_start:j])
                else:
                    feature_tau = tau_line[feature_start:j]
                    N_HI = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing

                if N_HI > 1e12:  # Only count above sensitivity threshold
                    column_densities.append(N_HI)

        # Handle case where feature extends to edge
        if in_feature:
            if colden_line is not None:
                N_HI = np.max(colden_line[feature_start:])
            else:
                feature_tau = tau_line[feature_start:]
                N_HI = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing
            if N_HI > 1e12:
                column_densities.append(N_HI)

    column_densities = np.array(column_densities)

    # Create histogram in log space
    if len(column_densities) > 0:
        log_N_min = 12.0  # log10(N_HI)
        log_N_max = 22.0
        n_bins = 50

        bins = np.logspace(log_N_min, log_N_max, n_bins)
        counts, bin_edges = np.histogram(column_densities, bins=bins)

        # Compute log-space bin properties for proper normalization
        log_bin_edges = np.log10(bin_edges)
        delta_log_N = np.diff(log_bin_edges)  # Constant for logspace bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        log_bin_centers = np.log10(bin_centers)

        # Fit power law in range 12 < log(N) < 14.5 (typical Lyman-alpha forest)
        # Adjusted range to focus on well-populated bins with MAX colden method
        fit_mask = (log_bin_centers > 12.0) & (log_bin_centers < 14.5)

        if np.sum(fit_mask) > 5 and np.sum(counts[fit_mask]) > 0:
            # Fit f(N) = A * N^-beta using properly normalized f(N)
            # f(N) in units of dN/dlog10(N)
            f_N_fit = counts[fit_mask] / delta_log_N[fit_mask]
            log_f_fit = np.log10(f_N_fit + 1e-10)  # Avoid log(0)
            log_N_fit = log_bin_centers[fit_mask]

            # Linear fit in log-log space
            valid = np.isfinite(log_f_fit) & (f_N_fit > 0)
            if np.sum(valid) > 2:
                coeffs = np.polyfit(log_N_fit[valid], log_f_fit[valid], 1)
                beta_fit = -coeffs[0]  # Negative slope
            else:
                beta_fit = np.nan
        else:
            beta_fit = np.nan
    else:
        bins = np.logspace(12, 22, 50)
        counts = np.zeros(len(bins) - 1)
        bin_edges = bins
        log_bin_edges = np.log10(bin_edges)
        delta_log_N = np.diff(log_bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        beta_fit = np.nan

    return {
        'N_HI': column_densities,
        'counts': counts,
        'bins': bin_edges,
        'bin_centers': bin_centers,
        'log_bin_edges': log_bin_edges,
        'delta_log_N': delta_log_N,
        'beta_fit': beta_fit,
        'n_absorbers': len(column_densities)
    }


# Compute effective optical depth tau_eff from flux.
def compute_effective_optical_depth(tau):
    flux = np.exp(-tau)

    # Global tau_eff
    mean_flux = np.mean(flux)
    tau_eff = -np.log(mean_flux)

    # Per-sightline tau_eff
    mean_flux_per_los = np.mean(flux, axis=1)
    tau_eff_per_los = -np.log(mean_flux_per_los)

    return {
        'tau_eff': float(tau_eff),
        'mean_flux': float(mean_flux),
        'tau_eff_per_sightline': tau_eff_per_los,
        'tau_eff_std': float(np.std(tau_eff_per_los)),
        'tau_eff_err': float(np.std(tau_eff_per_los) / np.sqrt(len(tau_eff_per_los)))
    }


# Compute line width (Doppler b-parameter) distribution from absorption features.
def compute_line_width_distribution(tau, velocity_spacing, threshold=0.5, colden=None):
    """See scripts/analysis.py for full documentation."""
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    # Physical constants
    m_H = 1.673e-24  # Hydrogen mass in grams
    k_B = 1.381e-16  # Boltzmann constant in CGS
    
    # Corrected constant
    TAU_TO_COLDEN_CONSTANT = 8.51e11  # cm^-2 / (km/s)

    column_densities = []
    b_parameters = []

    # Approximate Voigt profile for optical depth.
    def voigt_approx(v, tau_0, b, v_center):
        a = 4.7e-4  # Damping parameter for Lyman-alpha
        u = (v - v_center) / b

        # Gaussian core (approximation for thermal broadening)
        tau = tau_0 * np.exp(-u**2)
        return tau

    for i in range(tau.shape[0]):
        tau_line = tau[i, :]
        colden_line = colden[i, :] if colden is not None else None

        # Find peaks in optical depth (absorption features)
        peaks, properties = find_peaks(tau_line, height=threshold, distance=5)

        for peak_idx in peaks:
            # Define feature extent (where tau drops to threshold)
            left = peak_idx
            while left > 0 and tau_line[left] > threshold * 0.3:
                left -= 1

            right = peak_idx
            while right < len(tau_line) - 1 and tau_line[right] > threshold * 0.3:
                right += 1

            if right - left < 3:  # Too narrow to fit
                continue

            # Extract feature
            feature_tau = tau_line[left:right+1]
            feature_v = np.arange(len(feature_tau)) * velocity_spacing

            # Initial guess for Voigt fit
            tau_0_guess = tau_line[peak_idx]
            v_center_guess = (peak_idx - left) * velocity_spacing
            b_guess = 20.0  # km/s, typical IGM value

            try:
                # Fit Voigt profile
                popt, _ = curve_fit(
                    voigt_approx,
                    feature_v,
                    feature_tau,
                    p0=[tau_0_guess, b_guess, v_center_guess],
                    bounds=([0, 1.0, 0], [np.inf, 100.0, feature_v[-1]]),
                    maxfev=1000
                )

                tau_0_fit, b_fit, v_center_fit = popt

                # Estimate column density (use peak, not sum)
                if colden_line is not None:
                    N_HI = np.max(colden_line[left:right+1])
                else:
                    N_HI = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing

                # Only keep physically reasonable absorbers
                if N_HI > 1e12 and 2.0 < b_fit < 80.0:
                    column_densities.append(N_HI)
                    b_parameters.append(b_fit)

            except (RuntimeError, ValueError):
                # Fit failed, skip this feature
                continue

    column_densities = np.array(column_densities)
    b_parameters = np.array(b_parameters)

    # Convert b-parameters to temperatures (assuming thermal broadening)
    # b = sqrt(2kT/m) => T = (b^2 * m) / (2k)
    # For HI: T(K) = 1.28e4 * b(km/s)^2
    temperatures = 1.28e4 * b_parameters**2

    return {
        'N_HI': column_densities,
        'b_params': b_parameters,
        'temperatures': temperatures,
        'b_median': float(np.median(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'b_mean': float(np.mean(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'b_std': float(np.std(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'n_absorbers': len(b_parameters)
    }


# Compute temperature-density (T-ρ) relation from IGM gas along sightlines.
def compute_temperature_density_relation(temperature, density, tau, min_tau=0.1):
    # Flatten arrays and filter by optical depth
    temp_flat = temperature.flatten()
    dens_flat = density.flatten()
    tau_flat = tau.flatten()

    # Filter: only include absorbing gas (tau > min_tau) and valid values
    mask = (tau_flat > min_tau) & (temp_flat > 0) & (dens_flat > 0)
    mask &= np.isfinite(temp_flat) & np.isfinite(dens_flat)

    temp_filtered = temp_flat[mask]
    dens_filtered = dens_flat[mask]

    if len(temp_filtered) < 100:
        print(f"  Warning: Only {len(temp_filtered)} valid pixels for T-ρ fit")
        return {
            'temperature': temp_filtered,
            'density': dens_filtered,
            'log_T': np.array([]),
            'log_rho': np.array([]),
            'T0': np.nan,
            'gamma': np.nan,
            'gamma_err': np.nan,
            'n_pixels': len(temp_filtered)
        }

    # Convert density to overdensity (ρ/ρ_mean)
    rho_mean = np.median(dens_filtered)
    overdensity = dens_filtered / rho_mean

    # Take logarithms for power-law fit
    log_T = np.log10(temp_filtered)
    log_rho = np.log10(overdensity)

    # Fit T-ρ relation: log(T) = log(T0) + (gamma-1) * log(ρ/ρ_mean)
    # Robust fit using median binning
    rho_bins = np.linspace(log_rho.min(), log_rho.max(), 30)
    T_median = []
    rho_centers = []

    for i in range(len(rho_bins) - 1):
        mask_bin = (log_rho >= rho_bins[i]) & (log_rho < rho_bins[i+1])
        if np.sum(mask_bin) > 10:
            T_median.append(np.median(log_T[mask_bin]))
            rho_centers.append((rho_bins[i] + rho_bins[i+1]) / 2)

    if len(rho_centers) > 5:
        # Linear fit in log-log space
        coeffs = np.polyfit(rho_centers, T_median, 1)
        gamma_minus_1 = coeffs[0]
        log_T0 = coeffs[1]

        T0 = 10**log_T0
        gamma = gamma_minus_1 + 1.0

        # Estimate uncertainty
        T_pred = np.polyval(coeffs, rho_centers)
        residuals = np.array(T_median) - T_pred
        gamma_err = np.std(
            residuals) / np.std(rho_centers) if len(rho_centers) > 1 else np.nan
    else:
        T0 = np.nan
        gamma = np.nan
        gamma_err = np.nan

    return {
        'temperature': temp_filtered,
        'density': overdensity,
        'log_T': log_T,
        'log_rho': log_rho,
        'T0': float(T0) if np.isfinite(T0) else np.nan,
        'gamma': float(gamma) if np.isfinite(gamma) else np.nan,
        'gamma_err': float(gamma_err) if np.isfinite(gamma_err) else np.nan,
        'n_pixels': len(temp_filtered),
        'rho_mean': float(rho_mean)
    }


# Compute statistics for metal line absorption systems.
def compute_metal_line_statistics(tau, velocity_spacing, ion_name='Metal', threshold=0.05, colden=None):
    """See scripts/analysis.py for full documentation."""
    n_sightlines, n_pixels = tau.shape

    # Covering fraction: fraction of pixels with detectable absorption
    covering_fraction = np.sum(tau > threshold) / tau.size

    # Mean/median tau in absorbing regions
    tau_absorbing = tau[tau > threshold]
    if len(tau_absorbing) > 0:
        mean_tau_abs = np.mean(tau_absorbing)
        median_tau_abs = np.median(tau_absorbing)
    else:
        mean_tau_abs = 0.0
        median_tau_abs = 0.0

    # Count absorption systems
    column_densities = []
    n_systems = 0

    for i in range(n_sightlines):
        tau_line = tau[i, :]
        colden_line = colden[i, :] if colden is not None else None

        # Find absorption features
        absorbing = tau_line > threshold

        # Count connected regions
        in_feature = False
        feature_start = 0

        for j in range(len(tau_line)):
            if absorbing[j] and not in_feature:
                # Start of new feature
                in_feature = True
                feature_start = j
                n_systems += 1
            elif not absorbing[j] and in_feature:
                # End of feature
                in_feature = False
                
                # Estimate column density (use peak, not sum)
                if colden_line is not None:
                    N_ion = np.max(colden_line[feature_start:j])
                else:
                    feature_tau = tau_line[feature_start:j]
                    N_ion = 1e13 * np.sum(feature_tau) * velocity_spacing
                column_densities.append(N_ion)

        # Handle case where feature extends to edge
        if in_feature:
            if colden_line is not None:
                N_ion = np.max(colden_line[feature_start:])
            else:
                feature_tau = tau_line[feature_start:]
                N_ion = 1e13 * np.sum(feature_tau) * velocity_spacing
            column_densities.append(N_ion)

    column_densities = np.array(column_densities)

    # Compute dN/dz (line density per unit redshift)
    # Assuming each sightline samples ~dz = 0.1 (rough estimate)
    # For more accurate dN/dz, need actual path length
    dN_dz = n_systems / n_sightlines / 0.1 if n_sightlines > 0 else 0.0

    # Equivalent width (in velocity space, convert to Angstroms later)
    # EW ~ integral (1 - exp(-tau)) dv
    flux = np.exp(-tau)
    equivalent_width_vel = np.sum(1.0 - flux) * velocity_spacing  # km/s

    return {
        'ion_name': ion_name,
        'n_absorbers': n_systems,
        'n_sightlines': n_sightlines,
        'dN_dz': float(dN_dz),
        'covering_fraction': float(covering_fraction),
        'mean_tau': float(mean_tau_abs),
        'median_tau': float(median_tau_abs),
        # Per sightline
        'equivalent_width_vel': float(equivalent_width_vel / n_sightlines),
        'column_densities': column_densities,
        'log_N_mean': float(np.log10(np.mean(column_densities))) if len(column_densities) > 0 else np.nan,
        'log_N_median': float(np.log10(np.median(column_densities))) if len(column_densities) > 0 else np.nan
    }


# Create comparison plot for multiple spectral lines.
def plot_multi_line_comparison(line_stats_list, redshift, output_path, title=None):
    if len(line_stats_list) == 0:
        print("  Warning: No line statistics provided")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ion_names = [stats['ion_name'] for stats in line_stats_list]
    n_ions = len(ion_names)
    from matplotlib import cm
    cmap = cm.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, n_ions)]

    # Panel 1: Number of absorbers (dN/dz)
    ax = axes[0, 0]
    dN_dz_values = [stats['dN_dz'] for stats in line_stats_list]
    bars = ax.bar(range(n_ions), dN_dz_values, color=colors,
                  edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('dN/dz (absorbers per unit redshift)', fontsize=12)
    ax.set_title('Absorber Line Density', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Covering fraction
    ax = axes[0, 1]
    covering_fractions = [stats['covering_fraction']
                          * 100 for stats in line_stats_list]
    bars = ax.bar(range(n_ions), covering_fractions,
                  color=colors, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('Covering Fraction (%)', fontsize=12)
    ax.set_title('Sky Coverage', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Mean optical depth (in absorbing regions)
    ax = axes[1, 0]
    mean_taus = [stats['mean_tau'] for stats in line_stats_list]
    # Use log scale if range is large
    if max(mean_taus) / min([t for t in mean_taus if t > 0] + [1]) > 100:
        ax.set_yscale('log')
    bars = ax.bar(range(n_ions), mean_taus, color=colors,
                  edgecolor='black', alpha=0.7)
    ax.set_xticks(range(n_ions))
    ax.set_xticklabels(ion_names, rotation=45, ha='right')
    ax.set_ylabel('Mean tau (absorbing regions)', fontsize=12)
    ax.set_title('Absorption Strength', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Column density distributions (overlaid histograms)
    ax = axes[1, 1]
    for i, stats in enumerate(line_stats_list):
        if len(stats['column_densities']) > 0:
            log_N = np.log10(stats['column_densities'])
            ax.hist(log_N, bins=20, alpha=0.5, label=stats['ion_name'],
                    color=colors[i], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('log_10(N / cm^-2)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Column Density Distributions',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Overall title
    if title is None:
        title = f'Multi-Line Absorption Comparison - Redshift z = {
            redshift:.2f}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create flux power spectrum plot.
def plot_flux_power_spectrum(power_dict, redshift, output_path, title=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    k = power_dict['k']
    P_k = power_dict['P_k_mean']
    P_k_err = power_dict['P_k_err']

    # Only plot positive k (skip DC component)
    mask = k > 0
    k = k[mask]
    P_k = P_k[mask]
    P_k_err = P_k_err[mask]

    # Compute k*P(k)/pi following Khaire et al. (2019) convention
    kPk_pi = k * P_k / np.pi
    kPk_pi_err = k * P_k_err / np.pi

    # Plot with error bars
    ax.loglog(k, kPk_pi, 'o-', color='steelblue', linewidth=2,
              markersize=4, label=f'z = {redshift:.2f}')
    ax.fill_between(k, kPk_pi - kPk_pi_err, kPk_pi + kPk_pi_err,
                    alpha=0.3, color='steelblue')

    # Formatting
    ax.set_xlabel(r'Wavenumber $k$ [s/km]', fontsize=14)
    ax.set_ylabel(r'$k \cdot P_F(k) / \pi$ [dimensionless]', fontsize=14)
    ax.set_xlim(k[1], k[-1])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)

    if title is None:
        title = f'Lyman-α Flux Power Spectrum (z={redshift:.2f})'
    ax.set_title(title, fontsize=15, fontweight='bold')

    # Add info box
    info_text = f"N_sightlines = {power_dict['n_sightlines']}\n"
    info_text += f"Mean flux = {power_dict['mean_flux']:.3f}"
    ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create column density distribution plot.
def plot_column_density_distribution(cddf_dict, redshift, output_path, title=None):
    fig, ax = plt.subplots(figsize=(10, 7))

    bins = cddf_dict['bins']
    counts = cddf_dict['counts']
    bin_centers = cddf_dict['bin_centers']
    beta = cddf_dict['beta_fit']

    # Normalize to get f(N_HI) in units of dN/dlog10(N)
    # Use log-space bin widths for proper normalization
    if 'delta_log_N' in cddf_dict:
        delta_log_N = cddf_dict['delta_log_N']
    else:
        # Fallback for backward compatibility
        log_bin_edges = np.log10(bins)
        delta_log_N = np.diff(log_bin_edges)
    
    f_N = counts / delta_log_N

    # Plot
    mask = f_N > 0
    ax.loglog(bin_centers[mask], f_N[mask], 'o-', color='coral',
              linewidth=2, markersize=5, label='Measured')

    # Add power law fit if available
    if not np.isnan(beta):
        fit_range = (np.log10(bin_centers) > 13) & (np.log10(bin_centers) < 17)
        N_fit = bin_centers[fit_range]
        # Normalize to data at N ~ 1e14
        norm_idx = np.argmin(np.abs(bin_centers - 1e14))
        if f_N[norm_idx] > 0:
            A_norm = f_N[norm_idx] / (bin_centers[norm_idx]**(-beta))
            f_fit = A_norm * N_fit**(-beta)

            ax.loglog(N_fit, f_fit, '--', color='red', linewidth=2,
                      label=f'Power law: β = {beta:.2f}')

    # Formatting
    ax.set_xlabel(r'Column Density $N_{\rm HI}$ [cm$^{-2}$]', fontsize=14)
    ax.set_ylabel(r'$f(N_{\rm HI})$ [dN/d log$_{10}$ N]', fontsize=14)
    ax.set_xlim(1e12, 1e22)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=12)

    if title is None:
        title = f'Column Density Distribution (z={redshift:.2f})'
    ax.set_title(title, fontsize=15, fontweight='bold')

    # Add info box
    info_text = f"N_absorbers = {cddf_dict['n_absorbers']}\n"
    if not np.isnan(beta):
        info_text += f"β = {beta:.2f}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='top')

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create line width (b-parameter) distribution plot.
def plot_line_width_distribution(lwd_dict, redshift, output_path, title=None):
    if lwd_dict['n_absorbers'] == 0:
        print(f"  Warning: No absorbers found for line width analysis")
        return

    fig = plt.figure(figsize=(14, 6))

    # Create two subplots: b histogram and b(N_HI) correlation
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    b_params = lwd_dict['b_params']
    N_HI = lwd_dict['N_HI']

    # ===== Left panel: b-parameter histogram =====
    ax1.hist(b_params, bins=30, color='steelblue',
             alpha=0.7, edgecolor='black')
    ax1.axvline(lwd_dict['b_median'], color='red', linestyle='--', linewidth=2,
                label=f"Median = {lwd_dict['b_median']:.1f} km/s")
    ax1.axvline(lwd_dict['b_mean'], color='orange', linestyle='--', linewidth=2,
                label=f"Mean = {lwd_dict['b_mean']:.1f} km/s")

    ax1.set_xlabel('Doppler b-parameter (km/s)', fontsize=13)
    ax1.set_ylabel('Count', fontsize=13)
    ax1.set_title(f'Line Width Distribution (z = {
                  redshift:.2f})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add temperature scale on top
    ax1_top = ax1.twiny()
    b_ticks = np.array([5, 10, 20, 30, 40])
    T_ticks = 1.28e4 * b_ticks**2
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(b_ticks)
    ax1_top.set_xticklabels([f'{T/1e3:.0f}' for T in T_ticks])
    ax1_top.set_xlabel('Temperature (10³ K)', fontsize=12)

    # ===== Right panel: b(N_HI) correlation =====
    # Bin by column density for cleaner visualization
    log_N = np.log10(N_HI)
    N_bins = np.linspace(12, 17, 20)
    b_median_binned = []
    b_16th = []
    b_84th = []
    N_centers = []

    for i in range(len(N_bins) - 1):
        mask = (log_N >= N_bins[i]) & (log_N < N_bins[i+1])
        if np.sum(mask) > 5:
            b_in_bin = b_params[mask]
            b_median_binned.append(np.median(b_in_bin))
            b_16th.append(np.percentile(b_in_bin, 16))
            b_84th.append(np.percentile(b_in_bin, 84))
            N_centers.append((N_bins[i] + N_bins[i+1]) / 2)

    # Plot individual points (transparency for density)
    ax2.scatter(log_N, b_params, alpha=0.1, s=10,
                color='gray', label='Individual')

    # Plot binned median with scatter
    if len(N_centers) > 0:
        N_centers = np.array(N_centers)
        b_median_binned = np.array(b_median_binned)
        b_16th = np.array(b_16th)
        b_84th = np.array(b_84th)

        ax2.plot(N_centers, b_median_binned, 'o-', color='red', linewidth=2,
                 markersize=8, label='Median (binned)')
        ax2.fill_between(N_centers, b_16th, b_84th, alpha=0.3, color='red',
                         label='16-84th percentile')

    ax2.set_xlabel('log_10(N_HI / cm^-2)', fontsize=13)
    ax2.set_ylabel('Doppler b-parameter (km/s)', fontsize=13)
    ax2.set_title(
        f'b-N_HI Correlation (z = {redshift:.2f})', fontsize=14, fontweight='bold')
    ax2.set_xlim(12, 17)
    ax2.set_ylim(0, 60)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Overall title
    if title is None:
        title = f'Line Width Analysis - Redshift z = {redshift:.2f}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Add info box
    info_text = f"N_absorbers = {lwd_dict['n_absorbers']}\n"
    info_text += f"⟨b⟩ = {lwd_dict['b_mean']                          :.1f} ± {lwd_dict['b_std']:.1f} km/s\n"
    if lwd_dict['n_absorbers'] > 0:
        T_mean = 1.28e4 * lwd_dict['b_mean']**2
        info_text += f"⟨T⟩ = {T_mean/1e3:.0f} × 10³ K"

    ax1.text(0.95, 0.95, info_text, transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10, verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Create temperature-density relation plot.
def plot_temperature_density_relation(tdens_dict, redshift, output_path, title=None):
    if tdens_dict['n_pixels'] < 100:
        print(
            f"  Warning: Insufficient data for T-ρ plot ({tdens_dict['n_pixels']} pixels)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    from matplotlib.colors import LogNorm
    
    T_median = tdens_dict.get('T_median')
    rho_centers = tdens_dict.get('rho_centers')
    counts_per_bin = tdens_dict.get('counts_per_bin')
    T0 = tdens_dict.get('T0')
    gamma = tdens_dict.get('gamma')
    
    rho_valid = None
    T_valid = None
    counts_valid = None

    if T_median is not None and rho_centers is not None and len(T_median) > 0:
        valid_mask = np.isfinite(T_median) & np.isfinite(rho_centers)
        valid_mask &= (counts_per_bin > 0) if counts_per_bin is not None else True
        
        rho_valid = rho_centers[valid_mask]
        T_valid = T_median[valid_mask]
        counts_valid = counts_per_bin[valid_mask] if counts_per_bin is not None else None
        
        if len(rho_valid) > 0:
            scatter = ax.scatter(rho_valid, T_valid, c=counts_valid, 
                                s=100, cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=counts_valid.max() if counts_valid is not None else None),
                                edgecolors='black', linewidths=0.5, zorder=5)
            if counts_valid is not None:
                cbar = plt.colorbar(scatter, ax=ax, label='Count per bin')
            
            for i in range(len(rho_valid)):
                ax.annotate(f'{int(counts_valid[i]) if counts_valid is not None else ""}', 
                           (rho_valid[i], T_valid[i]), 
                           textcoords="offset points", xytext=(0, 10), 
                           ha='center', fontsize=8)

    if np.isfinite(T0) and np.isfinite(gamma):
        if rho_valid is not None and len(rho_valid) > 0:
            rho_range = np.linspace(rho_valid.min(), rho_valid.max(), 100)
        else:
            rho_range = np.linspace(-2, 2, 100)
        T_fit = np.log10(T0) + (gamma - 1) * rho_range
        ax.plot(rho_range, T_fit, 'b--', linewidth=3,
                label=f'T = T_0(rho/rho_bar)^(gamma-1)')

        fit_text = f'T_0 = {T0:.0f} K\ngamma = {gamma:.3f}'
        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12, verticalalignment='top', fontweight='bold')

    ax.set_xlabel('log_10(rho/rho_bar)', fontsize=14)
    ax.set_ylabel('log_10(T / K)', fontsize=14)

    if title is None:
        title = f'Temperature-Density Relation - Redshift z = {redshift:.2f}'
    ax.set_title(title, fontsize=15, fontweight='bold')

    if np.isfinite(gamma):
        ax.legend(fontsize=11, loc='lower right')

    ax.grid(True, alpha=0.3, linestyle='--')

    info_text = f"N_pixels = {tdens_dict['n_pixels']:,}\n"
    info_text += f"z = {redshift:.3f}"
    ax.text(0.95, 0.05, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout()
    save_plot(fig, output_path)
    plt.close()


# Format statistics dictionary as a nice text table.
def format_stats_table(stats):
    lines = [
        "Statistics:",
        "=" * 40,
        f"Mean flux:              {stats['mean_flux']:.4f}",
        f"Median flux:            {stats['median_flux']:.4f}",
        f"Std dev:                {stats['std_flux']:.4f}",
        "",
        f"Effective tau_eff:        {stats['effective_tau']:.4f}",
        f"Mean tau:                 {stats['mean_tau']:.4f}",
        f"Median tau:               {stats['median_tau']:.4f}",
        "",
        f"Deep absorption:        {stats['deep_absorption_frac']*100:.2f}%",
        f"Moderate absorption:    {
            stats['moderate_absorption_frac']*100:.2f}%",
        f"Weak absorption:        {stats['weak_absorption_frac']*100:.2f}%",
    ]
    return "\n".join(lines)


# =============================== #
# SIMULATION COMPARISON FRAMEWORK #
# =============================== #

# Load and compute all analysis results from a spectra file.
def load_spectra_results(spectra_file, velocity_spacing=0.1):
    import h5py

    results = {
        'filepath': str(spectra_file),
        'success': False,
    }

    try:
        with h5py.File(spectra_file, 'r') as f:
            # Get redshift
            redshift = None
            if 'Header' in f:
                header = f['Header'].attrs
                redshift = header.get('redshift', header.get('Redshift', None))
            results['redshift'] = float(
                redshift) if redshift is not None else None

            # Auto-detect tau data (try HI Lya first)
            tau = None
            tau_path = None
            if 'tau/H/1/1215' in f:
                tau = np.array(f['tau/H/1/1215'])
                tau_path = 'tau/H/1/1215'
            elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
                tau = np.array(f['tau'])
                tau_path = 'tau'

            if tau is None:
                results['error'] = 'No tau data found'
                return results

            flux = np.exp(-tau)
            n_sightlines, n_pixels = tau.shape

            results['n_sightlines'] = n_sightlines
            results['n_pixels'] = n_pixels

            # Compute all analyses
            results['flux_stats'] = compute_flux_statistics(tau)
            results['tau_eff'] = compute_effective_optical_depth(tau)
            results['power_spectrum'] = compute_power_spectrum(
                flux, velocity_spacing)
            results['cddf'] = compute_column_density_distribution(
                tau, velocity_spacing)

            # Try line width analysis
            try:
                results['line_widths'] = compute_line_width_distribution(
                    tau, velocity_spacing)
            except:
                results['line_widths'] = None

            # Try T-ρ analysis
            try:
                if tau_path and '/' in tau_path:
                    parts = tau_path.split('/')
                    temp_elem = parts[1] if len(parts) >= 2 else 'H'
                    temp_ion = parts[2] if len(parts) >= 3 else '1'
                else:
                    temp_elem, temp_ion = 'H', '1'

                has_temp = ('temperature' in f and temp_elem in f['temperature'] and
                            temp_ion in f['temperature'][temp_elem])
                has_dens = ('density_weight_density' in f and temp_elem in f['density_weight_density'] and
                            temp_ion in f['density_weight_density'][temp_elem])

                if has_temp and has_dens:
                    temperature = np.array(
                        f['temperature'][temp_elem][temp_ion])
                    density = np.array(
                        f['density_weight_density'][temp_elem][temp_ion])
                    results['temp_density'] = compute_temperature_density_relation(
                        temperature, density, tau, min_tau=0.1
                    )
                else:
                    results['temp_density'] = None
            except:
                results['temp_density'] = None

            results['success'] = True

    except Exception as e:
        results['error'] = str(e)
        return results

    return results


# Compare results from multiple simulation runs.
def compare_simulations(spectra_files, labels=None, output_path=None):
    if labels is None:
        labels = [f"Sim {i}" for i in range(len(spectra_files))]

    # Load all results
    print(f"Loading {len(spectra_files)} simulation results...")
    all_results = []
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        print(f"[{i+1}/{len(spectra_files)}] {label}: {fpath}")
        results = load_spectra_results(fpath)
        if results['success']:
            results['label'] = label
            all_results.append(results)
            print(f"\tz={results['redshift']:.3f}, N={
                  results['n_sightlines']}")
        else:
            print(f"\tFailed: {results.get('error', 'unknown error')}")

    if len(all_results) == 0:
        print("Error: No valid results to compare")
        return None

    # Create comparison figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Panel 1: Flux power spectrum
    ax = fig.add_subplot(gs[0, :])
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k = ps['k']
        P_k = ps['P_k_mean']
        P_k_err = ps['P_k_err']

        mask = k > 0
        k = k[mask]
        P_k = P_k[mask]
        P_k_err = P_k_err[mask]

        color = colors[i % len(colors)]
        ax.loglog(k, P_k, 'o-', color=color, linewidth=2, markersize=3,
                  label=f"{res['label']} (z={res['redshift']:.2f})", alpha=0.8)
        ax.fill_between(k, P_k - P_k_err, P_k +
                        P_k_err, alpha=0.2, color=color)

    ax.set_xlabel(r'Wavenumber $k$ [s/km]', fontsize=13)
    ax.set_ylabel(r'Power Spectrum $P_F(k)$ [km/s]', fontsize=13)
    ax.set_title('Flux Power Spectrum Comparison',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')

    # Panel 2: Effective optical depth
    ax = fig.add_subplot(gs[1, 0])
    tau_effs = [res['tau_eff']['tau_eff'] for res in all_results]
    tau_errs = [res['tau_eff']['tau_eff_err'] for res in all_results]
    x_pos = np.arange(len(all_results))

    bars = ax.bar(x_pos, tau_effs, yerr=tau_errs, capsize=5,
                  color=[colors[i % len(colors)]
                         for i in range(len(all_results))],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([res['label']
                       for res in all_results], rotation=45, ha='right')
    ax.set_ylabel(r'Effective Optical Depth $\tau_{\rm eff}$', fontsize=12)
    ax.set_title('Effective Optical Depth', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Mean flux
    ax = fig.add_subplot(gs[1, 1])
    mean_fluxes = [res['flux_stats']['mean_flux'] for res in all_results]

    bars = ax.bar(x_pos, mean_fluxes,
                  color=[colors[i % len(colors)]
                         for i in range(len(all_results))],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([res['label']
                       for res in all_results], rotation=45, ha='right')
    ax.set_ylabel(r'Mean Transmitted Flux $\langle F \rangle$', fontsize=12)
    ax.set_title('Mean Flux', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Column density distribution
    ax = fig.add_subplot(gs[2, 0])
    for i, res in enumerate(all_results):
        cddf = res['cddf']
        if cddf['n_absorbers'] > 0 and len(cddf['counts']) > 0:
            color = colors[i % len(colors)]
            
            # Properly compute log-space normalization
            bin_centers = cddf['bin_centers']
            log_bin_centers = np.log10(bin_centers)
            
            # Get delta_log_N from cddf_dict or compute it
            if 'delta_log_N' in cddf:
                delta_log_N = cddf['delta_log_N'][0]  # Constant for logspace
            else:
                log_bins = np.log10(cddf['bins'])
                delta_log_N = np.mean(np.diff(log_bins))
            
            # f(N) in units of dN/dlog10(N) per sightline
            f_N = cddf['counts'] / (cddf['n_absorbers'] * delta_log_N)

            # Only plot non-zero bins
            mask = f_N > 0
            if np.any(mask):
                ax.scatter(log_bin_centers[mask], f_N[mask],
                           s=30, alpha=0.6, color=color, label=res['label'])

                # Plot fit if available
                if not np.isnan(cddf['beta_fit']):
                    N_fit = np.logspace(12, 16, 100)
                    # Power law: f(N) = A * N^(-beta)
                    A_norm = f_N[mask].max() / (bin_centers[mask][np.argmax(f_N[mask])]**(-cddf['beta_fit']))
                    f_fit = A_norm * N_fit**(-cddf['beta_fit'])
                    ax.plot(np.log10(N_fit), f_fit, '--',
                            color=color, alpha=0.5, linewidth=1.5)

    ax.set_xlabel(r'$\log_{10}(N_{\rm HI} / {\rm cm}^{-2})$', fontsize=12)
    ax.set_ylabel(r'$f(N_{\rm HI})$ [dN/d log$_{10}$ N]', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Column Density Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 5: Temperature-density parameters (if available)
    ax = fig.add_subplot(gs[2, 1])
    T0_values = []
    gamma_values = []
    valid_labels = []

    for i, res in enumerate(all_results):
        if res['temp_density'] is not None:
            td = res['temp_density']
            if np.isfinite(td['T0']) and np.isfinite(td['gamma']):
                T0_values.append(td['T0'])
                gamma_values.append(td['gamma'])
                valid_labels.append(res['label'])

    if len(T0_values) > 0:
        x_pos_td = np.arange(len(T0_values))

        ax2 = ax.twinx()

        bars1 = ax.bar(x_pos_td - 0.2, T0_values, width=0.4,
                       color='steelblue', alpha=0.7, label=r'$T_0$ [K]')
        bars2 = ax2.bar(x_pos_td + 0.2, gamma_values, width=0.4,
                        color='coral', alpha=0.7, label=r'$\gamma$')

        ax.set_xticks(x_pos_td)
        ax.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax.set_ylabel(
            r'$T_0$ at Mean Density [K]', fontsize=12, color='steelblue')
        ax2.set_ylabel(r'Polytropic Index $\gamma$',
                       fontsize=12, color='coral')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.set_title('IGM Equation of State', fontsize=13, fontweight='bold')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=9, loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No temperature-density\ndata available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Overall title
    fig.suptitle('Simulation Comparison', fontsize=16,
                 fontweight='bold', y=0.995)

    if output_path:
        save_plot(fig, output_path)
        print(f"\nComparison plot saved: {output_path}")

    plt.close()

    # Return summary
    comparison = {
        'n_simulations': len(all_results),
        'labels': [res['label'] for res in all_results],
        'results': all_results
    }

    return comparison


# =========================== #
# REDSHIFT EVOLUTION TRACKING #
# =========================== #

# Track how observables evolve with redshift across multiple snapshots.
def track_redshift_evolution(spectra_files, labels=None, output_path=None):
    if labels is None:
        labels = [f"Snap {i}" for i in range(len(spectra_files))]

    # Load all results
    print(f"Loading {len(spectra_files)} snapshots for evolution tracking...")
    all_results = []
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        print(f"[{i+1}/{len(spectra_files)}] {label}: {fpath}")
        results = load_spectra_results(fpath)
        if results['success'] and results['redshift'] is not None:
            results['label'] = label
            all_results.append(results)
            print(f"z={results['redshift']:.3f}")
        else:
            print(f"Failed or no redshift")

    if len(all_results) == 0:
        print("Error: No valid results for evolution tracking")
        return None

    # Sort by redshift
    all_results.sort(key=lambda x: x['redshift'])

    # Extract evolution data
    redshifts = np.array([res['redshift'] for res in all_results])
    tau_effs = np.array([res['tau_eff']['tau_eff'] for res in all_results])
    tau_errs = np.array([res['tau_eff']['tau_eff_err'] for res in all_results])
    mean_fluxes = np.array([res['flux_stats']['mean_flux']
                           for res in all_results])
    n_absorbers = np.array([res['cddf']['n_absorbers'] for res in all_results])

    # T-ρ parameters (if available)
    T0_values = []
    gamma_values = []
    z_with_T = []

    for res in all_results:
        if res['temp_density'] is not None:
            td = res['temp_density']
            if np.isfinite(td['T0']) and np.isfinite(td['gamma']):
                z_with_T.append(res['redshift'])
                T0_values.append(td['T0'])
                gamma_values.append(td['gamma'])

    z_with_T = np.array(z_with_T)
    T0_values = np.array(T0_values)
    gamma_values = np.array(gamma_values)

    # Create evolution figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Effective optical depth evolution
    ax = axes[0, 0]
    ax.errorbar(redshifts, tau_effs, yerr=tau_errs, fmt='o-',
                color='steelblue', linewidth=2, markersize=6, capsize=4, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Effective Optical Depth $\tau_{\rm eff}$', fontsize=13)
    ax.set_title(r'$\tau_{\rm eff}(z)$ Evolution',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Higher z on left

    # Panel 2: Mean flux evolution
    ax = axes[0, 1]
    ax.plot(redshifts, mean_fluxes, 'o-', color='coral',
            linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Mean Transmitted Flux $\langle F \rangle$', fontsize=13)
    ax.set_title(r'$\langle F \rangle(z)$ Evolution',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Panel 3: Number of absorbers
    ax = axes[1, 0]
    ax.plot(redshifts, n_absorbers, 'o-', color='green',
            linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel('Number of Absorbers', fontsize=13)
    ax.set_title('Absorber Count Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Panel 4: Temperature-density evolution
    ax = axes[1, 1]
    if len(z_with_T) > 0:
        ax2 = ax.twinx()

        line1 = ax.plot(z_with_T, T0_values, 'o-', color='steelblue',
                        linewidth=2, markersize=6, alpha=0.8, label=r'$T_0$')
        line2 = ax2.plot(z_with_T, gamma_values, 's-', color='coral',
                         linewidth=2, markersize=6, alpha=0.8, label=r'$\gamma$')

        ax.set_xlabel('Redshift $z$', fontsize=13)
        ax.set_ylabel(
            r'$T_0$ at Mean Density [K]', fontsize=13, color='steelblue')
        ax2.set_ylabel(r'Polytropic Index $\gamma$',
                       fontsize=13, color='coral')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.set_title('IGM Equation of State Evolution',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # Combined legend
        lines = line1 + line2
        labels_leg = [l.get_label() for l in lines]
        ax.legend(lines, labels_leg, fontsize=11, loc='best')
    else:
        ax.text(0.5, 0.5, 'No temperature-density\ndata available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Overall title
    z_min, z_max = redshifts.min(), redshifts.max()
    fig.suptitle(f'Redshift Evolution (z = {z_min:.2f} - {z_max:.2f})',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if output_path:
        save_plot(fig, output_path)
        print(f"\nEvolution plot saved: {output_path}")

    plt.close()

    # Return summary
    evolution = {
        'n_snapshots': len(all_results),
        'redshift_range': (float(redshifts.min()), float(redshifts.max())),
        'redshifts': redshifts.tolist(),
        'tau_eff': tau_effs.tolist(),
        'mean_flux': mean_fluxes.tolist(),
        'n_absorbers': n_absorbers.tolist(),
        'results': all_results
    }

    if len(z_with_T) > 0:
        evolution['T0'] = T0_values.tolist()
        evolution['gamma'] = gamma_values.tolist()
        evolution['z_with_T'] = z_with_T.tolist()

    return evolution


# =============================== #
# VPFIT INTEGRATION WITH VOIGTFIT #
# =============================== #

def compute_column_density_distribution_vpfit(flux, wavelength, redshift, threshold=0.05,
                                              continuum_window=50, min_snr=5.0,
                                              output_dir=None):
    try:
        import VoigtFit
        from astropy import units as u
        import astropy.constants as const
    except ImportError:
        print("ERROR: VoigtFit not installed. Install with: pip install VoigtFit")
        return {'error': 'VoigtFit not available'}

    # Convert wavelength to velocity space (km/s) relative to Lyman-alpha
    lambda_lya = 1215.67  # Angstroms
    lambda_rest = lambda_lya * (1 + redshift)

    # Convert wavelength to velocity (km/s)
    c = 299792.458  # km/s
    velocity = c * (wavelength - lambda_rest) / lambda_rest

    # Convert flux to optical depth
    tau = -np.log(flux)

    column_densities = []
    b_parameters = []
    absorber_redshifts = []
    errors_N = []
    errors_b = []
    chi_squared_values = []

    # Process each sightline
    for i in range(flux.shape[0]):
        flux_line = flux[i, :]
        tau_line = tau[i, :]

        # Find absorption features
        absorbers = _find_absorption_features(tau_line, velocity, threshold)

        for absorber in absorbers:
            try:
                # Fit Voigt profile to this absorber
                result = _fit_single_absorber(flux_line, wavelength, absorber,
                                              redshift, continuum_window, min_snr)

                if result is not None:
                    column_densities.append(result['N_HI'])
                    b_parameters.append(result['b'])
                    absorber_redshifts.append(result['z_abs'])
                    errors_N.append(result['N_err'])
                    errors_b.append(result['b_err'])
                    chi_squared_values.append(result['chi2'])

            except Exception as e:
                print(f"Warning: Failed to fit absorber in sightline {i}: {e}")
                continue

    # Convert to numpy arrays
    column_densities = np.array(column_densities)
    b_parameters = np.array(b_parameters)
    absorber_redshifts = np.array(absorber_redshifts)
    errors_N = np.array(errors_N)
    errors_b = np.array(errors_b)
    chi_squared_values = np.array(chi_squared_values)

    return {
        'N_HI': column_densities,
        'b_params': b_parameters,
        'redshifts': absorber_redshifts,
        'errors_N': errors_N,
        'errors_b': errors_b,
        'chi_squared': chi_squared_values,
        'n_absorbers': len(column_densities),
        'method': 'VoigtFit'
    }


# Find absorption features in optical depth spectrum.
def _find_absorption_features(tau, velocity, threshold):
    absorbers = []

    # Find contiguous regions above threshold
    in_feature = False
    start_idx = 0

    for i in range(len(tau)):
        if tau[i] > threshold and not in_feature:
            # Start of feature
            in_feature = True
            start_idx = i
        elif tau[i] <= threshold and in_feature:
            # End of feature
            end_idx = i - 1

            # Check if feature is significant
            if end_idx - start_idx >= 3:  # At least 3 pixels
                absorbers.append({
                    'vel_start': velocity[start_idx],
                    'vel_end': velocity[end_idx],
                    'tau_max': np.max(tau[start_idx:end_idx+1]),
                    'pixel_start': start_idx,
                    'pixel_end': end_idx
                })

            in_feature = False

    # Handle feature at end
    if in_feature and len(tau) - start_idx >= 3:
        absorbers.append({
            'vel_start': velocity[start_idx],
            'vel_end': velocity[-1],
            'tau_max': np.max(tau[start_idx:]),
            'pixel_start': start_idx,
            'pixel_end': len(tau) - 1
        })

    return absorbers


# Fit a single absorber with VoigtFit.
def _fit_single_absorber(flux, wavelength, absorber, redshift, continuum_window, min_snr):
    try:
        from VoigtFit import VoigtFitter
        import astropy.units as u

        # Extract region around absorber
        pixel_start = max(0, absorber['pixel_start'] - continuum_window)
        pixel_end = min(len(flux), absorber['pixel_end'] + continuum_window)

        flux_region = flux[pixel_start:pixel_end]
        wave_region = wavelength[pixel_start:pixel_end]

        # Create VoigtFit spectrum object
        spec = VoigtFitter(wave_region, flux_region)

        # Estimate continuum (simple linear fit to edges)
        edge_size = min(10, len(flux_region)//4)
        left_flux = np.mean(flux_region[:edge_size])
        right_flux = np.mean(flux_region[-edge_size:])
        continuum = np.linspace(left_flux, right_flux, len(flux_region))
        spec.set_continuum(continuum)

        # Estimate initial parameters
        center_vel = (absorber['vel_start'] + absorber['vel_end']) / 2
        center_wave = 1215.67 * (1 + redshift) * (1 + center_vel/299792.458)

        # Rough column density estimate from optical depth
        tau_max = absorber['tau_max']
        dv = np.abs(absorber['vel_end'] - absorber['vel_start'])
        N_est = 1.13e14 * tau_max * dv  # Simplified estimate

        # Rough b estimate from width
        b_est = dv / 2.355  # FWHM to sigma conversion

        # Add HI Lyman-alpha component
        spec.add_line('HI1215', center_wave, N_est, b_est, z=redshift)

        # Fit the spectrum
        spec.fit()

        # Extract results
        if spec.lines:
            line = spec.lines[0]  # First (and only) line

            # Check fit quality
            chi2 = spec.chi2
            if chi2 > 100:  # Poor fit
                return None

            return {
                'N_HI': line.N.value,
                'b': line.b.value,
                'z_abs': line.z,
                'N_err': line.N.std if hasattr(line.N, 'std') else 0.1,
                'b_err': line.b.std if hasattr(line.b, 'std') else 5.0,
                'chi2': chi2
            }

    except Exception as e:
        print(f"VoigtFit error: {e}")
        return None


# ============================== #
# TEMPERATURE/DENSITY COMPUTATION #
# ============================== #

def compute_temp_density_chunked(spec, elem, ion, chunk_size=None, verbose=True):
    """
    Compute temperature and density-weighted density in chunks to reduce memory usage.
    Each sightline is computed independently, so chunking produces identical results
    to computing all at once, but with much lower peak memory usage.
    """
    import gc
    
    # Get total number of sightlines
    n_sightlines = spec.cofm.shape[0]
    
    # Auto-detect chunk size if not specified
    if chunk_size is None:
        if n_sightlines < 1000:
            chunk_size = n_sightlines  # No chunking for small datasets
        elif n_sightlines < 5000:
            chunk_size = 1000
        else:
            chunk_size = 2000
    
    n_chunks = (n_sightlines + chunk_size - 1) // chunk_size
    
    if verbose:
        if n_chunks == 1:
            print(f"Computing temperature and density for {n_sightlines} sightlines...")
        else:
            print(f"\nComputing temperature and density in {n_chunks} chunks of ~{chunk_size} sightlines...")
            print(f"  Total sightlines: {n_sightlines}")
            print(f"  Chunk size: {chunk_size}")
    
    # Save original sightline configuration
    original_cofm = spec.cofm
    original_axis = spec.axis
    original_numlos = spec.NumLos
    
    # Save original colden (needed for density normalization)
    original_colden = spec.colden.get((elem, ion))
    if original_colden is None:
        return False, f"Column density not found for {elem} {ion}. Must compute tau/colden first."
    
    temp_chunks = []
    dens_chunks = []
    
    try:
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_sightlines)
            n_in_chunk = end_idx - start_idx
            
            if verbose and n_chunks > 1:
                print(f"\n  Chunk {i+1}/{n_chunks}: sightlines {start_idx}-{end_idx-1} ({n_in_chunk} sightlines)")
            
            # Garbage collection before each chunk
            gc.collect()
            
            # Temporarily modify spec to use subset of sightlines
            spec.cofm = original_cofm[start_idx:end_idx]
            spec.axis = original_axis[start_idx:end_idx]
            spec.NumLos = n_in_chunk
            spec.colden[(elem, ion)] = original_colden[start_idx:end_idx]
            
            # Compute temperature for this chunk
            if verbose:
                prefix = "    " if n_chunks > 1 else ""
                print(f"{prefix}Computing temperature...", end=' ', flush=True)
            try:
                temp_chunk = spec._get_mass_weight_quantity(spec._temp_single_file, elem, ion)
                temp_chunks.append(temp_chunk)
                if verbose:
                    print(f"OK (shape: {temp_chunk.shape})")
            except Exception as e:
                return False, f"Temperature computation failed on chunk {i+1}: {e}"
            
            # Clear memory between temp and density
            del temp_chunk
            gc.collect()
            
            # Compute density for this chunk
            if verbose:
                prefix = "    " if n_chunks > 1 else ""
                print(f"{prefix}Computing density-weighted density...", end=' ', flush=True)
            try:
                dens_chunk = spec._get_mass_weight_quantity(spec._densweightdens, elem, ion)
                dens_chunks.append(dens_chunk)
                if verbose:
                    print(f"OK (shape: {dens_chunk.shape})")
            except Exception as e:
                return False, f"Density computation failed on chunk {i+1}: {e}"
            
            # Clear memory after chunk
            del dens_chunk
            gc.collect()
        
        # Restore original configuration
        spec.cofm = original_cofm
        spec.axis = original_axis
        spec.NumLos = original_numlos
        spec.colden[(elem, ion)] = original_colden
        
        # Concatenate all chunks
        if verbose and n_chunks > 1:
            print(f"\n  Concatenating {n_chunks} chunks...")
        
        temp_full = np.concatenate(temp_chunks, axis=0)
        dens_full = np.concatenate(dens_chunks, axis=0)
        
        if verbose and n_chunks > 1:
            print(f"    Temperature shape: {temp_full.shape}")
            print(f"    Density shape: {dens_full.shape}")
        
        # Store in spec object
        spec.temp[(elem, ion)] = temp_full
        spec.dens_weight_dens[(elem, ion)] = dens_full
        
        # Final cleanup
        del temp_chunks, dens_chunks, temp_full, dens_full
        gc.collect()
        
        if verbose and n_chunks > 1:
            print("  ✓ Temperature and density computation complete")
        
        return True, None
        
    except Exception as e:
        # Always restore original configuration on error
        spec.cofm = original_cofm
        spec.axis = original_axis
        spec.NumLos = original_numlos
        spec.colden[(elem, ion)] = original_colden
        
        return False, f"Unexpected error: {e}"
