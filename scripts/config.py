import os
import re
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR = PROJECT_ROOT / "output"
SPECTRA_DIR = PROJECT_ROOT / "spectra"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# Analysis output subdirectories
SIGHTLINES_DIR = OUTPUT_DIR / "sightlines"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"

# Ensure output directories exist
PLOTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SPECTRA_DIR.mkdir(exist_ok=True)
SIGHTLINES_DIR.mkdir(exist_ok=True)
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)
COMPARISON_DIR.mkdir(exist_ok=True)


# Optical depth above which a pixel counts as "absorbing" when identifying
# contiguous absorption features. Governs the b-parameter / line-width
# distribution and the metal lines; the CDDF uses fixed cells and needs no
# threshold.
#
# WARNING: TAU_THRESHOLD_HI = 0.5 is an inherited constant with no published
# justification. It is not a neutral choice: tau > 0.5 means F < 0.607, so for
# an unsaturated Lya line (tau_0 = 7.58e-8 N_HI / b) the absorber catalogue is
# effectively empty below log N_HI ~ 13.0 (b = 15 km/s) to 13.4 (b = 40 km/s).
# At z >~ 3 the opposite problem dominates: contiguous tau > 0.5 runs merge
# across the saturated forest and are counted as one absorber -- at z = 6 every
# sightline collapses to a single feature at every threshold tested. Treat it as
# a systematic to be quoted, not a hidden default.
TAU_THRESHOLD_HI = 0.5
TAU_THRESHOLD_METAL = 0.05
TAU_THRESHOLD_VPFIT = 0.05


# --------------------------------------------------------------------------
# CDDF production configuration
# --------------------------------------------------------------------------
#   absorber_mode = 2 (fixed 50 km/s cells)
#       Needs no tau threshold, and is half as sensitive to the fit window as
#       per-feature deblending (beta spread 0.19-0.24 vs 0.50). Identical
#       bin-for-bin to fake_spectra's line=False. The whole-sightline mode
#       (line=True) is unusable at this box size -- beta = -1.8 at z = 2.
#
#   colden_mode = 1 (sum, not max)
#       colden is a PER-PIXEL column density, so the reduction over an absorber
#       must be a sum. max censors the high-N tail, which shows up as a beta
#       nearly independent of both threshold and fit window.
#
#   dx_mode = 1 (absorption distance X(z)), norm_mode = 1 (per linear dN)
#       Together these give f(N) = dn/dN dX in cm^2, the quantity the literature
#       quotes. A z-independent comoving dX puts a spurious (1+z)^2 into the
#       CDDF's redshift evolution; dividing by dex fits beta_true - 1.
#
#   grid 13.0 -> 22.8 in 49 bins == fake_spectra's default 0.2 dex grid.
#   min_N_gate = 0 -- fake_spectra applies no low-N gate.
#
#   fit window [13.5, 15.0]
#       Defensible only because cells removed the threshold. A floor below ~13.5
#       sits inside the completeness turnover; a ceiling above ~15 runs out of
#       populated bins.
CDDF_OPTIONS = {
    'absorber_mode': 2,
    'cell_dv': 50.0,
    'colden_mode': 1,
    'dx_mode': 1,
    'norm_mode': 1,
    'log_N_min': 13.0,
    'log_N_max': 22.8,
    'n_bins': 49,
    'fit_log_N_min': 13.5,
    'fit_log_N_max': 15.0,
    'min_N_gate': 0.0,
}

# Above this redshift the Lya forest is saturated and no absorber definition
# recovers a physical CDDF slope (at z = 6, 50 km/s cells give beta = -3.3).
# beta_fit is reported as NaN there rather than as a number that would silently
# enter a plot. tau_eff and the flux PDF are unaffected, as neither depends on
# absorber identification.
CDDF_SATURATED_REDSHIFT = 4.5


# CAMEL simulation suites available
CAMEL_SUITES = {
    'IllustrisTNG': DATA_DIR / "IllustrisTNG",
    'SIMBA': DATA_DIR / "SIMBA",
    'Astrid': DATA_DIR / "Astrid",
}


def get_available_datasets(suite='IllustrisTNG'):
    datasets = {}

    if suite not in CAMEL_SUITES:
        return datasets

    suite_path = CAMEL_SUITES[suite]

    if not suite_path.exists():
        return datasets

    # Look for set directories (e.g., LH, 1P, CV, etc.)
    for set_dir in suite_path.iterdir():
        if not set_dir.is_dir():
            continue

        sim_set = set_dir.name

        # Look inside each set directory for simulation directories (e.g., LH_0, LH_1, etc.)
        for sim_dir in set_dir.iterdir():
            if not sim_dir.is_dir():
                continue

            dir_name = sim_dir.name

            # Extract simulation identifier from directory name
            # Strategy: Remove the prefix "{sim_set}_" to get the identifier
            # For LH: "LH_80" -> extract "80"
            # For 1P: "1P_p11_2" -> extract "p11_2", "1P_0" -> extract "0"
            # This handles any pattern: {SET}_{IDENTIFIER}
            
            sim_identifier = None
            
            if dir_name.startswith(f"{sim_set}_"):
                # Remove the prefix to get the identifier
                sim_identifier = dir_name[len(sim_set) + 1:]  # +1 for underscore

            if sim_identifier:
                # Verify this directory contains simulation data
                # Check for snapshots, spectra files, or SPECTRA subdirectories
                has_data = (list(sim_dir.glob("snap_*.hdf5")) or
                            list(sim_dir.glob("camel_*_spectra_snap_*.hdf5")) or
                            list(sim_dir.glob("SPECTRA_*")))

                if has_data:
                    if sim_set not in datasets:
                        datasets[sim_set] = []
                    datasets[sim_set].append(sim_identifier)

    # Sort the simulation numbers for each set
    # Custom sort: pure numbers first (sorted numerically), then alphanumeric (sorted lexically)
    for sim_set in datasets:
        def sort_key(x):
            # If it's a pure number, sort numerically first
            if x.isdigit():
                return (0, int(x), '')
            # Otherwise, sort alphabetically second
            else:
                return (1, 0, x)
        
        datasets[sim_set] = sorted(datasets[sim_set], key=sort_key)

    return datasets


def get_available_snapshots(suite='IllustrisTNG', sim_set='LH', sim_num='0'):
    snapshots = {}

    if suite not in CAMEL_SUITES:
        return {}

    sim_name = sim_set + '_' + sim_num
    suite_path = CAMEL_SUITES[suite] / sim_set / sim_name

    if not suite_path.exists():
        return snapshots

    # Look for snapshot files in simulation directory
    for snap_file in suite_path.glob("snap_*.hdf5"):
        snap_num = int(snap_file.stem.split('_')[-1])
        snapshots[snap_num] = snap_file

    # Look for spectra files directly in simulation directory
    for spectra_file in suite_path.glob("camel_*_spectra_snap_*.hdf5"):
        # Extract snapshot number from filename
        parts = spectra_file.stem.split('_')
        if 'snap' in parts:
            snap_idx = parts.index('snap')
            if snap_idx + 1 < len(parts):
                try:
                    # Get the part after 'snap', which should be the number
                    snap_part = parts[snap_idx + 1]
                    # Handle case where there might be additional suffixes like 'n100'
                    snap_num = int(snap_part)
                    # Store as spectra file (distinguish from snapshot)
                    key = f"spectra_{snap_num}"
                    snapshots[key] = spectra_file
                except ValueError:
                    continue

    # Also check SPECTRA_* subdirectories for generated spectra
    for spectra_dir in suite_path.glob("SPECTRA_*"):
        for spectra_file in spectra_dir.glob("camel_*_spectra_snap_*.hdf5"):
            # Extract snapshot number from filename
            parts = spectra_file.stem.split('_')
            if 'snap' in parts:
                snap_idx = parts.index('snap')
                if snap_idx + 1 < len(parts):
                    try:
                        snap_part = parts[snap_idx + 1]
                        snap_num = int(snap_part)
                        # Store as spectra file (distinguish from snapshot)
                        key = f"spectra_{snap_num}"
                        snapshots[key] = spectra_file
                    except ValueError:
                        continue

    return snapshots


def extract_simulation_info(filepath):
    filepath = Path(filepath)
    parts = filepath.parts
    
    if 'spectra' in parts:
        spectra_idx = parts.index('spectra')
        suite = parts[spectra_idx + 1] if spectra_idx + 1 < len(parts) else 'Unknown'
        sim_set = parts[spectra_idx + 2] if spectra_idx + 2 < len(parts) else 'Unknown'
        sim_name = parts[spectra_idx + 3] if spectra_idx + 3 < len(parts) else 'Unknown'
        
        stem = filepath.stem
        if 'snap_' in stem:
            snap_parts = stem.split('snap_')
            if len(snap_parts) > 1:
                snap_num = snap_parts[1].split('_')[0]
            else:
                snap_num = 'unknown'
        else:
            snap_num = 'unknown'
    
    elif 'data' in parts or len(parts) >= 4:
        suite = parts[-4] if len(parts) >= 4 else 'Unknown'
        sim_set = parts[-3] if len(parts) >= 3 else 'Unknown'
        sim_name = parts[-2] if len(parts) >= 2 else 'Unknown'
        snap_num = filepath.stem.split('_')[-1]
    
    else:
        suite = 'Unknown'
        sim_set = 'Unknown'
        sim_name = 'Unknown'
        snap_num = 'unknown'
    
    return {
        'suite': suite,
        'sim_set': sim_set,
        'sim_name': sim_name,
        'snap_num': snap_num
    }


def find_snapshot_file(spectra_file, data_root=None):
    info = extract_simulation_info(spectra_file)

    snap_num = info['snap_num']
    if snap_num == 'unknown':
        match = re.search(r'snap[_-](\d+)', Path(spectra_file).name)
        if not match:
            return None
        snap_num = match.group(1)

    root = Path(data_root) if data_root is not None else DATA_DIR
    spectra_dir = Path(spectra_file).parent

    candidates = []
    if info['suite'] in CAMEL_SUITES:
        candidates.append(root / info['suite'] / info['sim_set'] /
                          info['sim_name'] / f'snap_{snap_num}.hdf5')
    candidates += [
        spectra_dir / f'snap_{snap_num}.hdf5',
        spectra_dir / f'snap-{snap_num}.hdf5',
        root / f'snap_{snap_num}.hdf5',
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def get_spectra_output_dir(snapshot_path, spectra_type='camel'):
    info = extract_simulation_info(snapshot_path)
    
    output_dir = SPECTRA_DIR / info['suite'] / info['sim_set'] / info['sim_name']
    
    if spectra_type == 'cgm':
        output_dir = output_dir / 'cgm'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


# Print summary of all available simulation data
def list_all_available_data():
    print("=" * 70)
    print("AVAILABLE SIMULATION DATA")
    print("=" * 70)

    for suite_name, suite_path in CAMEL_SUITES.items():
        if not suite_path.exists():
            continue

        print(f"\n{suite_name}:")

        # Dynamically discover all available datasets
        datasets = get_available_datasets(suite_name)

        if not datasets:
            print(f"  No datasets found")
            continue

        for sim_set in sorted(datasets.keys()):
            print(f"  {sim_set}:")

            for sim_num in datasets[sim_set]:
                snapshots = get_available_snapshots(
                    suite_name, sim_set, sim_num)

                if not snapshots:
                    continue

                # Separate simulation snapshots from spectra files
                sim_snaps = {k: v for k, v in snapshots.items()
                             if isinstance(k, int)}
                spectra_files = {k: v for k, v in snapshots.items()
                                 if isinstance(k, str)}

                if sim_snaps or spectra_files:
                    print(f"    {sim_set}_{sim_num}:")

                    if sim_snaps:
                        snap_nums = sorted(sim_snaps.keys())
                        print(f"      Snapshots: {snap_nums}")

                    if spectra_files:
                        spectra_nums = sorted([int(k.split('_')[1])
                                              for k in spectra_files.keys()])
                        print(f"      Spectra:   {spectra_nums}")

    print("=" * 70)


# =========================== #
# DEFAULT ANALYSIS PARAMETERS #
# =========================== #

# Spectra generation parameters
DEFAULT_NUM_SIGHTLINES = 100        # Number of random sightlines
DEFAULT_PIXEL_RESOLUTION = 0.1      # Resolution in km/s
# Size of simulation box in Mpc/h (for CAMEL)
DEFAULT_BOX_SIZE = 25.0
DEFAULT_RANDOM_SEED = None          # Random seed for reproducibility

# Wavelength range for Lyman-alpha (in Angstroms)
LYMAN_ALPHA_WAVELENGTH = 1215.67    # Rest wavelength
WAVELENGTH_MIN = 1180.0             # Minimum wavelength
WAVELENGTH_MAX = 1250.0             # Maximum wavelength

# Redshift parameters
REDSHIFT_MIN = 2.0                  # Minimum redshift of interest
REDSHIFT_MAX = 6.0                  # Maximum redshift of interest

# Analysis parameters
FLUX_BINS = 50                      # Number of bins for flux histograms
TAU_MAX = 10.0                      # Maximum optical depth to plot
POWER_SPECTRUM_KMIN = 0.001         # Minimum k for power spectrum (s/km)
POWER_SPECTRUM_KMAX = 0.1           # Maximum k for power spectrum (s/km)


# Spectral line definitions: (element, ion_level, wavelength_angstrom, name)
# Format used by fake_spectra: spec.get_tau(element, ion_level, wavelength)
SPECTRAL_LINES = {
    'lya':  ('H',  1, 1215, 'Lyman-alpha'),
    # ion = -1: fake_spectra skips the get_reproc_HI() multiplication
    # (spectra.py:597) and treats all hydrogen as neutral, i.e. the Gunn-Peterson
    # optical depth. Saved under tau/H/-1/1215 and colden/H/-1.
    'lya_h': ('H', -1, 1215, 'Lyman-alpha (total H, no ionisation correction)'),
    'lyb':  ('H',  1, 1025, 'Lyman-beta'),
    'heii': ('He', 2, 303,  'HeII-303'),
    'civ':  ('C',  4, 1548, 'CIV-1548'),
    'ovi':  ('O',  6, 1031, 'OVI-1031'),
    'mgii': ('Mg', 2, 2796, 'MgII-2796'),
    'siiv': ('Si', 4, 1393, 'SiIV-1393'),
}


# Get spectral line information
def get_line_info(line_code):
    return SPECTRAL_LINES.get(line_code.lower())


# Parse comma-separated list of spectral lines
def parse_line_list(line_string):
    lines = [l.strip().lower() for l in line_string.split(',')]

    # Validate all lines exist
    invalid = [l for l in lines if l not in SPECTRAL_LINES]
    if invalid:
        return None

    return lines


# PLOTTING CONFIGURATION
# Figure sizes (width, height in inches)
FIGSIZE_SINGLE = (10, 6)            # Single panel plots
FIGSIZE_DOUBLE = (14, 6)            # Two-panel plots
FIGSIZE_QUAD = (12, 10)             # Four-panel plots

# DPI for saved figures
PLOT_DPI = 300

# Color schemes
COLORS_QUALITATIVE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Line styles
LINESTYLE_OBSERVED = '--'
LINESTYLE_SIMULATED = '-'
LINEWIDTH_DEFAULT = 1.5

# Font sizes
FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 12
FONTSIZE_TICK = 10
FONTSIZE_LEGEND = 10


# Convert comoving length to proper length
def comoving_to_proper(comoving_length, scale_factor):
    return comoving_length * scale_factor


# Convert redshift to scale factor
def redshift_to_scale_factor(redshift):
    return 1.0 / (1.0 + redshift)


# Convert scale factor to redshift
def scale_factor_to_redshift(scale_factor):
    return (1.0 / scale_factor) - 1.0


# FILE NAMING CONVENTIONS
# Generate standard output filename for spectra
def get_snapshot_output_name(snapshot_path, lines=None, num_sightlines=None):
    snapshot_path = Path(snapshot_path)

    # Extract snapshot number
    snap_num = None
    parts = snapshot_path.stem.split('_')
    for i, part in enumerate(parts):
        if part == 'snap' and i + 1 < len(parts):
            snap_num = parts[i + 1]
            break

    if snap_num is None:
        snap_num = 'unknown'

    if lines is None:
        lines = ['lya']  # Default
    elif isinstance(lines, str):
        lines = [lines]

    # Build filename with line codes
    line_str = '_'.join(lines)
    filename = f"camel_{line_str}_spectra_snap_{snap_num}"

    if num_sightlines is not None:
        filename += f"_n{num_sightlines}"

    filename += ".hdf5"

    return filename


# Generate standard plot filename
def get_plot_output_name(snapshot_path, plot_type, extension='png'):
    snapshot_path = Path(snapshot_path)
    
    info = extract_simulation_info(snapshot_path)
    
    plot_dir = PLOTS_DIR / info['suite'] / info['sim_set'] / info['sim_name']
    
    # If this is a CGM spectra file, add cgm subdirectory
    if 'cgm' in snapshot_path.parts:
        plot_dir = plot_dir / 'cgm'
    
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    filename = f"camel_{plot_type}_snap_{info['snap_num']}.{extension}"
    
    return plot_dir / filename


def print_config_summary():
    """
    Print summary of current configuration
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Data directory:   {DATA_DIR}")
    print(f"Plots directory:  {PLOTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Default Parameters:")
    print(f"  Sightlines:     {DEFAULT_NUM_SIGHTLINES}")
    print(f"  Resolution:     {DEFAULT_PIXEL_RESOLUTION} km/s")
    print(f"  Box size:       {DEFAULT_BOX_SIZE} Mpc/h")
    print(f"  Random seed:    {DEFAULT_RANDOM_SEED}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_config_summary()
    list_all_available_data()
