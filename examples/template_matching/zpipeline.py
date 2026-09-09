"""Transit search pipeline on the calibrated matched-filter S/N (z).

The steps follow Section 2 of the paper:

1. Download the TESS light curves of the target (SPOC 2-min; QLP optional).
2. Normalisation model w_n: a running median of length WINDOW_DAYS smoothed
   by a boxcar of the same length (bruce.data.median_filter + convolve_1d).
3. Template width: radius_1 optimised on the reference event with k held at
   a placeholder (z is invariant to the template depth).
4. Scan every sector with the fixed template at trial epochs W/20 apart
   (bruce.template_match.template_match_snr): S = 2 dlogL, standardised to
   z and MAD-whitened; nothing is fitted per epoch.
5. Catalogue peaks above the local false-alarm threshold (P_LOCAL, or a hard
   Z_THRESHOLD), at least one transit width apart.
6. Refit the depth (k) at each catalogued peak only, width held fixed; grid
   panels are drawn at each peak's own fitted depth.

Outputs: zresults_<tic>.pkl ({"template", "results", "global_data"}) and the
snr_/grid_ plots, written to the working directory and stamped with the
target name and date.  Helpers for a multiple-event search, phase
dispersion, ephemeris fitting and folding are included for the follow-up.

Run:  python zpipeline.py   (edit the USER INPUTS block below)

Requires this repository's bruce (template_match_snr and the FAP helpers)
plus numpy, scipy, matplotlib and astropy.
"""

import math
import os
import pickle
import time as clock
import warnings
from datetime import date
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
from astropy.units import UnitsWarning

import bruce

warnings.filterwarnings("ignore", category=UnitsWarning)

VERSION = "2026-09-09d"

# ======================================================
# USER INPUTS
# ======================================================

# Target
TIC_ID = 350618622
TARGET_NAME = "TOI-201"  # used in output file names
REFERENCE_SECTOR = 2  # sector holding the reference event
EPOCH = 21.88  # days after the reference-sector start; None = whole sector

# Template
INITIAL_RADIUS_1 = 0.035  # starting width parameter for the optimiser
INITIAL_K = 0.08  # placeholder depth for the scan (z is depth-invariant);
# the depth is fitted per catalogued peak afterwards
PERIOD = 30  # assumed orbital period for the template (days)
STEP = None  # trial-epoch spacing (days); None = W/20

# Normalisation model: running median + boxcar, both of length WINDOW_DAYS.
# Set it per target: at least 3 x the transit width, so the median never sees
# the transit as the majority of its window, and shorter than the star's
# variability timescale.  1 day is bruce's own default.  On TOI-201
# (W = 4.9 h) 0.6 d was the most sensitive, 1.0 d costs ~20 per cent of z
# and 1.2 d a third.
WINDOW_DAYS = 1.0

# Detection threshold: local (per trial epoch) false-alarm probability.
# z_p = Phi^-1(1 - P_LOCAL): 1e-4 -> 3.72, 1e-6 -> 4.75, 1e-8 -> 5.61,
# 1e-9 -> 6.00.  Recommended for real data: 1e-9.  TESS scans carry
# non-Gaussian systematics in the z ~ 5-6 band; 1e-9 sits just above it
# while leaving real events untouched (TOI-201: all chain transits survive).
# The expected number of false peaks per scan (~ 5T/W x P_LOCAL) is printed.
P_LOCAL = 1e-9

# Hard threshold override: a z value (e.g. 6.0) catalogues peaks above it
# directly, ignoring P_LOCAL; the implied local FAP is reported.
Z_THRESHOLD = None

# Edge masking: drop trial epochs within EDGE_MASK_WIDTHS transit widths of a
# series edge or of a gap longer than SEGMENT_GAP_DAYS, where the baseline is
# least constrained.  ON suppresses edge-overhang peaks; OFF keeps sensitivity
# to transits partially covered at the edges.  On TOI-201, masking >= 0.5 W
# already costs real transits.  The multiple-event search always masks edges.
MASK_EDGES = False
EDGE_MASK_WIDTHS = 1.0
SEGMENT_GAP_DAYS = 0.2

EXCLUDE_QLP = True  # QLP FFI systematics defeat the per-scan whitening
TRANSIT_WINDOW = 0.5  # minimum half-width (days) of the transit cut-outs

Z_RESULTS_FILE = f"zresults_{TIC_ID}.pkl"


# ======================================================
# Data loading and pre-processing
# ======================================================

def load_datasets(tic_id, use_ffi=False, retries=3, retry_wait=30,
                  reuse_saved=True):
    """Download TESS light curves; returns (summary, datasets, labels, path).

    bruce's loader fetches every sector afresh on each run (it moves the
    files out of astroquery's cache), so a dropped MAST connection aborts
    the whole run.  The download is retried `retries` times, `retry_wait`
    seconds apart; if it still fails and a results pickle from a previous
    run of this target exists (Z_RESULTS_FILE), the raw sector arrays saved
    in it are reused -- they are the same PDCSAP data the loader returns.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return bruce.ambiguous_period.download_tess_data(tic_id, use_ffi=use_ffi)
        except Exception as exc:  # network / MAST failures
            last_error = exc
            print(f"MAST download failed (attempt {attempt}/{retries}): {exc}")
            if attempt < retries:
                clock.sleep(retry_wait)
    if reuse_saved and os.path.exists(Z_RESULTS_FILE):
        print(f"Reusing the sector arrays saved in {Z_RESULTS_FILE}")
        with open(Z_RESULTS_FILE, "rb") as fh:
            saved = pickle.load(fh)
        datasets = [SimpleNamespace(time=np.asarray(r["time"], float),
                                    flux=np.asarray(r["flux"], float),
                                    flux_err=np.asarray(r["flux_err"], float))
                    for r in saved["results"]]
        labels = [r["label"] for r in saved["results"]]
        return None, datasets, labels, os.path.abspath(Z_RESULTS_FILE)
    raise last_error


def get_reference_dataset(datasets, labels, reference_sector):
    """Return the dataset whose label matches "Sector <n>" exactly."""
    for dataset, label in zip(datasets, labels):
        if f"Sector {reference_sector} " in f"{label} ":
            return dataset
    raise ValueError(f"No such sector {reference_sector}")


def resolve_window(window_days, width_days):
    """Check the manual flatten window (days) against the transit width."""
    if window_days is None:
        raise ValueError("WINDOW_DAYS must be set (days); keep it >= 3 x the "
                         "transit width and shorter than the star's variability")
    window_days = float(window_days)
    if window_days < 2.0 * float(width_days):
        warnings.warn(f"flatten window {window_days:.2f} d is under twice the "
                      f"transit width ({width_days * 24:.1f} h): the median "
                      "filter will follow the transit; use >= 3 W.")
    return window_days


def flatten_lightcurve(time, flux, flux_err, window_days):
    """Estimate the out-of-transit baseline w_n; returns
    (flattened_flux, flattened_err, normalisation).

    Running median of length `window_days` smoothed by a boxcar of the same
    length (bruce.data.median_filter + convolve_1d), the paper's estimator
    (Section 2): the median is robust to the sparse in-transit points and
    absorbs several times less of the transit than a polynomial or
    moving-average fit.  The window is a per-target choice (WINDOW_DAYS).
    """
    t64 = np.ascontiguousarray(time, dtype=np.float64)
    f64 = np.ascontiguousarray(flux, dtype=np.float64)
    cadence = np.nanmedian(np.diff(t64))
    if window_days / cadence > 1024:
        # bruce's C median filter keeps at most 1024 points per window and
        # silently truncates the rest from the right-hand side
        warnings.warn(
            f"median flatten: window {window_days:.2f} d spans "
            f"{window_days / cadence:.0f} cadences but bruce's median_filter "
            "uses at most 1024 points per window (truncated one-sidedly); "
            "bin the light curve or shorten the window.")
    normalisation = bruce.data.median_filter(t64, f64, float(window_days))
    normalisation = bruce.data.convolve_1d(t64, normalisation, float(window_days))
    return flux / normalisation, flux_err / normalisation, normalisation


def pre_process(dataset, flatten_kwargs):
    """Flatten one dataset; returns a dict of raw and flattened arrays."""
    time, flux, flux_err = dataset.time, dataset.flux, dataset.flux_err
    flattened_flux, flattened_err, normalisation = flatten_lightcurve(
        time, flux, flux_err, **flatten_kwargs)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "flattened_flux": flattened_flux,
        "flattened_err": flattened_err,
        "normalisation": normalisation,
    }


def get_ref_window(reference_dataset, epoch, halfwidth=2):
    """(start, end) of a window centred `epoch` days into the reference sector."""
    transit_time = reference_dataset.time.min() + epoch
    return transit_time - halfwidth, transit_time + halfwidth


# ======================================================
# Template matching (calibrated z statistic)
# ======================================================

def template_width(radius_1, k, period):
    """Transit width in days for a central (b = 0) transit."""
    return bruce.binarystar.transit_width(radius_1, k, 0.0, period=period)


def radius_1_for_width(width_days, k, period):
    """Invert template_width for radius_1."""
    return brentq(lambda r1: template_width(r1, k, period) - width_days,
                  1e-4, 0.5, xtol=1e-6)


def z_template_match(time, flux, flux_err, normalisation, period, radius_1, k,
                     time_trial=None, step=0.005, whiten=True, rho2=None):
    """Scan the template; returns (time_trial, z, S, rho2)."""
    return bruce.template_match.template_match_snr(
        time, flux, flux_err,
        normalisation_model=normalisation,
        period=period,
        radius_1=radius_1,
        k=k,
        time_trial=time_trial,
        time_step=step,
        whiten=whiten,
        rho2=rho2
    )


def interior_mask(rho2):
    """Trial epochs where the template is well covered by data
    (rho2 above half its median).  Used only by the multiple-event
    search, where it is unconditional; peak cataloguing uses the
    geometric edge_epoch_mask instead (see MASK_EDGES)."""
    rho2 = np.asarray(rho2, dtype=float)
    good = rho2 > 1e-9
    mask = good.copy()
    if good.any():
        mask &= rho2 > 0.5 * np.median(rho2[good])
    return mask


def segment_edge_times(time_values, gap_days=0.2):
    """Start and end times of every contiguous data segment
    (bruce.template_match.segment_edge_times)."""
    return bruce.template_match.segment_edge_times(time_values, gap_days=gap_days)


def edge_epoch_mask(trial_time, time_values, mask_days, gap_days=0.2):
    """True for trial epochs at least ``mask_days`` from every data-segment
    edge (series ends and gaps > ``gap_days``); wraps
    bruce.template_match.edge_epoch_mask with the distance given in days
    (typically EDGE_MASK_WIDTHS transit widths)."""
    return bruce.template_match.edge_epoch_mask(trial_time, time_values, float(mask_days),
                                                mask_widths=1.0, gap_days=gap_days)


def whitening_scale(z, S, rho2):
    """Per-scan factor s such that the returned z equals z_raw / s, where
    z_raw = (S + rho2) / (2 rho) is the unwhitened statistic.  Equals 1 when
    the scan was run with whiten=False."""
    rho2 = np.asarray(rho2, dtype=float)
    S = np.asarray(S, dtype=float)
    z = np.asarray(z, dtype=float)
    ok = rho2 > 1e-9
    z_raw = np.zeros_like(rho2)
    z_raw[ok] = (S[ok] + rho2[ok]) / (2.0 * np.sqrt(rho2[ok]))
    good = ok & np.isfinite(z) & (np.abs(z) > 1e-12)
    if not good.any():
        return 1.0
    return float(np.nanmedian(z_raw[good] / z[good]))


def model_depth(radius_1, k, period, width_days):
    """Maximum fractional dip of the transit model (b = 0)."""
    tf = np.linspace(-0.6 * width_days, 0.6 * width_days, 1201)
    return float(1.0 - bruce.binarystar.lc(tf, t_zero=0.0, period=period,
                                           radius_1=radius_1, k=k).min())


def fit_peak_depth(time, flux, flux_err, normalisation, period, width_days,
                   t_peak, scale=1.0, k_bounds=(1e-3, 0.5)):
    """Fit the transit depth at one catalogued peak by optimising k.

    The scan itself fits nothing per epoch (z already profiles the depth,
    so detection gains nothing from it); this refit runs only at the
    catalogued peaks.  k is optimised on 2 dlogL with the template width
    held at `width_days` (radius_1 re-derived for every k) and the epoch
    fixed at `t_peak`, so it is a pure depth measurement that leaves the
    width and the detection untouched.  The error is the likelihood
    interval where 2 dlogL falls by scale^2 below its maximum, i.e. one
    sigma once the stated errors are rescaled by the per-scan whitening
    factor `scale` (see whitening_scale).

    Returns a dict: k, k_err_lo, k_err_hi, depth_ppm, depth_err_ppm,
    radius_1 (at the fitted k), S_max (2 dlogL at the fit) and at_bound
    (True when the optimum sits on a k bound and should not be trusted).

    The depth is measured against the normalisation model, i.e. it is the
    post-flatten depth that the grid panels display; a window that follows
    part of the transit lowers it (about 10 per cent at 3 W in simulations).
    For a physical depth, re-flatten with the event masked and refit.
    """
    t_arr = np.array([float(t_peak)], dtype=float)

    def S_of_k(k):
        r1 = radius_1_for_width(width_days, k, period)
        _, s = bruce.template_match.template_match_lightcurve(
            time, flux, flux_err, normalisation, period=period,
            radius_1=r1, k=k, time_trial=t_arr)
        return float(np.asarray(s)[0])

    res = minimize_scalar(lambda k: -S_of_k(k), bounds=k_bounds,
                          method="bounded", options={"xatol": 1e-5})
    k_hat, S_max = float(res.x), float(-res.fun)
    at_bound = bool(k_hat - k_bounds[0] < 1e-4 or k_bounds[1] - k_hat < 1e-4)

    target = S_max - float(scale) ** 2

    def crossing(lo, hi):
        try:
            return float(brentq(lambda k: S_of_k(k) - target, lo, hi, xtol=1e-6))
        except ValueError:
            return np.nan

    k_lo = crossing(k_bounds[0], k_hat) if S_of_k(k_bounds[0]) < target else np.nan
    k_hi = crossing(k_hat, k_bounds[1]) if S_of_k(k_bounds[1]) < target else np.nan

    r1_hat = radius_1_for_width(width_days, k_hat, period)
    depth = model_depth(r1_hat, k_hat, period, width_days)
    sides = []
    if np.isfinite(k_lo):
        sides.append(depth - model_depth(radius_1_for_width(width_days, k_lo, period),
                                         k_lo, period, width_days))
    if np.isfinite(k_hi):
        sides.append(model_depth(radius_1_for_width(width_days, k_hi, period),
                                 k_hi, period, width_days) - depth)
    depth_err = float(np.mean(sides)) if sides else np.nan

    return {
        "k": k_hat,
        "k_err_lo": float(k_hat - k_lo) if np.isfinite(k_lo) else np.nan,
        "k_err_hi": float(k_hi - k_hat) if np.isfinite(k_hi) else np.nan,
        "depth_ppm": depth * 1e6,
        "depth_err_ppm": depth_err * 1e6,
        "radius_1": r1_hat,
        "S_max": S_max,
        "at_bound": at_bound,
    }


def detect_peaks_snr(time_trial, z, rho2, width_days, p_local=1e-6,
                     mask_edges=False, mask_widths=1.0, data_time=None,
                     z_threshold=None, gap_days=0.2):
    """Peaks above the detection threshold, separated by >= one width.

    The threshold is z_p = Phi^-1(1 - p_local), or `z_threshold` directly
    when given (the implied local FAP Phi(-z) then drives the reporting).
    If `mask_edges` (and `data_time` is supplied), trial epochs within
    `mask_widths` transit widths of a series edge or a gap > `gap_days`
    are dropped; otherwise only zero-coverage epochs are excluded, keeping
    sensitivity to transits that overhang a data edge.

    Returns (indices, times, heights, threshold, expected_false), where
    expected_false ~ N_eff x p is the expected number of false peaks in
    this scan (N_eff = 5T/W, the MC-measured ~5 independent tests per
    transit width; see the bruce docstrings for provenance and caveats).
    """
    rho2 = np.asarray(rho2, dtype=float)
    keep = rho2 > 1e-9
    if mask_edges and data_time is not None:
        keep &= edge_epoch_mask(time_trial, data_time,
                                mask_widths * width_days, gap_days)

    if z_threshold is not None:
        threshold = float(z_threshold)
        p_eff = float(stats.norm.sf(threshold))
    else:
        _, thresholds = bruce.template_match.get_snr_height_from_fap(
            p_value=[p_local], n_independent=1)
        threshold = float(thresholds[0])
        p_eff = p_local
    n_eff = bruce.template_match.effective_independent_trials(
        time_trial[keep], width_days, decorrelation_factor=5.0)
    expected_false = 1.0 - (1.0 - p_eff) ** n_eff
    z = np.where(keep & np.isfinite(z), z, -np.inf)

    # pad so first/last trial epochs can register as peaks
    padded = np.concatenate([[-np.inf], z, [-np.inf]])
    raw, _ = find_peaks(padded, height=threshold)
    raw = raw - 1

    # keep the strongest of any cluster closer than one width
    ranked = raw[np.argsort(-z[raw], kind="stable")]
    kept = []
    for i in ranked:
        if all(abs(time_trial[i] - time_trial[j]) >= width_days for j in kept):
            kept.append(int(i))
    kept = np.array(sorted(kept, key=lambda i: time_trial[i]), dtype=int)

    return kept, time_trial[kept], z[kept], threshold, expected_false


def objective_z(radius_1, k, time, flux, flux_err, normalisation, period,
                step, time_trial, whiten=False):
    """Negative max z of a scan, for the width optimiser.

    Uses the raw matched-filter z by default.  The MAD whitening scale over
    a short reference window is inflated by the transit itself and by any
    smooth residual the flatten leaves, and it grows with the template
    width, so a whitened objective is biased towards narrow templates.
    """
    _, zz, _, _ = z_template_match(
        time, flux, flux_err, normalisation,
        period=period, radius_1=radius_1, k=k,
        time_trial=time_trial, step=step, whiten=whiten)
    return -np.max(zz)


def z_optimise_template(reference_dataset, period, initial_radius_1, initial_k,
                        epoch=None, step=None, window_days=None,
                        max_width_days=None):
    """Optimise the template width (radius_1) on the maximum raw z; k fixed.

    The search evaluates a coarse geometric grid of widths between the
    bounds first and then refines around the best grid point, so a narrow
    template locking on to an ingress cannot trap the optimiser (see
    objective_z for why the raw rather than whitened z is maximised).
    z is invariant to the template depth, so k cannot be fitted from max z;
    the depth is fitted afterwards at each catalogued peak (fit_peak_depth,
    called from z_run_template_matching).  The width search
    is capped at max_width_days (default window_days / 3) because the
    flatten leaves correlated wings that an unbounded fit climbs; if the
    fit lands on the cap, enlarge window_days rather than trusting the
    width.  If `epoch` is given the scan is restricted to a window around
    it.  `window_days` is the flatten window (WINDOW_DAYS); `step` None
    means the paper's W/20 trial-epoch spacing.
    """
    width0 = template_width(initial_radius_1, initial_k, period)
    if step is None:
        step = width0 / 20.0  # the paper's trial-epoch spacing
    window_days = resolve_window(window_days, width0)
    reference = pre_process(reference_dataset, {"window_days": window_days})
    time = reference["time"]
    flux = reference["flux"]
    flux_err = reference["flux_err"]
    normalisation = reference["normalisation"]

    reference_window = None
    time_trial = None
    if epoch is not None:
        reference_window = get_ref_window(reference_dataset, epoch)
        time_trial = np.arange(reference_window[0], reference_window[1], step)

    if max_width_days is None:
        max_width_days = window_days / 3.0
    cadence = np.nanmedian(np.diff(time))
    min_width_days = max(4.0 * cadence, 2.0 * step)
    r1_lo = radius_1_for_width(min_width_days, initial_k, period)
    r1_hi = radius_1_for_width(max_width_days, initial_k, period)

    args = (initial_k, time, flux, flux_err, normalisation, period,
            step, time_trial)
    grid = np.geomspace(r1_lo, r1_hi, 12)
    grid_vals = np.array([objective_z(r, *args) for r in grid])
    i_best = int(np.argmin(grid_vals))
    result = minimize_scalar(
        objective_z,
        bounds=(grid[max(i_best - 1, 0)], grid[min(i_best + 1, grid.size - 1)]),
        args=args,
        method="bounded"
    )
    if grid_vals[i_best] < result.fun:  # keep the grid point if the refinement did worse
        result.x, result.fun = grid[i_best], grid_vals[i_best]
    best_radius_1 = float(result.x)
    if template_width(best_radius_1, initial_k, period) > 0.95 * max_width_days:
        warnings.warn(
            f"optimised width sits at the search cap "
            f"({max_width_days * 24:.1f} h); increase window_days "
            f"(keep it >~ 6x the expected transit width) and re-run.")

    time_best, z_best, S_best, rho2_best = z_template_match(
        time, flux, flux_err, normalisation,
        period=period, radius_1=best_radius_1, k=initial_k,
        time_trial=time_trial, step=step)
    best_t0 = time_best[np.argmax(z_best)]

    return {
        "radius_1": best_radius_1,
        "k": initial_k,
        "t0": best_t0,
        "z": z_best,
        "S": S_best,
        "rho2": rho2_best,
        "time_trial": time_best,
        "normalisation": normalisation,
        "flattened_flux": reference["flattened_flux"],
        "flattened_err": reference["flattened_err"],
        "window_days": window_days,
        "optimisation": result,
        "reference_window": reference_window,
    }


def z_run_template_matching(datasets, labels, template, period, step=None,
                            window_days=None,
                            transit_window=0.5, p_local=1e-6, whiten=True,
                            mask_edges=False, mask_widths=1.0,
                            z_threshold=None, gap_days=0.2):
    """Scan every dataset with the template and catalogue significant peaks.

    `window_days` is the flatten window (WINDOW_DAYS); `step` None means
    the paper's W/20 trial-epoch spacing.
    Returns (results, global_data): per-dataset dicts with the scan, peaks
    and threshold, and the concatenated arrays used downstream.
    """
    width = template_width(template["radius_1"], template["k"], period)
    cut_halfwidth = max(transit_window, width)
    if step is None:
        step = width / 20.0  # the paper's trial-epoch spacing
    window_days = resolve_window(window_days, width)
    print(f"\nFlatten: running median + boxcar, window {window_days:.3f} d "
          f"({window_days / width:.1f} x the template width {width * 24:.2f} h)")

    results = []
    peak_number = 1
    offset = 0

    all_peak_indices, all_peak_times = [], []
    all_time, all_flux, all_flux_err = [], [], []
    all_labels, all_time_trial = [], []
    transit_time, transit_flux, transit_flux_err = [], [], []

    for dataset, label in zip(datasets, labels):
        processed = pre_process(dataset, {"window_days": window_days})
        time = processed["time"]
        flattened_flux = processed["flattened_flux"]
        flattened_err = processed["flattened_err"]

        print(f"\nStarting template matching for {label}...")
        time_trial, z, S, rho2 = z_template_match(
            time, processed["flux"], processed["flux_err"],
            processed["normalisation"],
            period=period,
            radius_1=template["radius_1"],
            k=template["k"],
            step=step,
            whiten=whiten
        )

        peak_idx, peak_times, peak_heights, threshold, expected_false = \
            detect_peaks_snr(time_trial, z, rho2, width, p_local=p_local,
                             mask_edges=mask_edges, mask_widths=mask_widths,
                             data_time=time, z_threshold=z_threshold,
                             gap_days=gap_days)

        # Depth refit at the catalogued peaks only (the scan itself fits
        # nothing per epoch): k on 2 dlogL with the width held at the
        # template width and the epoch fixed at the peak.
        scale = whitening_scale(z, S, rho2)
        peak_fits = [fit_peak_depth(time, processed["flux"], processed["flux_err"],
                                    processed["normalisation"], period, width, pt,
                                    scale=scale)
                     for pt in peak_times]

        peak_numbers = []
        for pt in peak_times:
            mask = np.abs(time - pt) < cut_halfwidth
            transit_time.append(time[mask])
            transit_flux.append(flattened_flux[mask])
            transit_flux_err.append(flattened_err[mask])

            all_peak_times.append(pt)
            peak_numbers.append(peak_number)
            peak_number += 1
        all_peak_indices.extend(peak_idx + offset)

        all_time.append(time)
        all_flux.append(flattened_flux)
        all_flux_err.append(flattened_err)
        all_labels.append(label)
        all_time_trial.append(time_trial)
        offset += len(time_trial)

        if z_threshold is not None:
            detail = f"z override; implied local FAP {stats.norm.sf(threshold):.2g}"
        else:
            detail = f"local FAP {p_local:g}"
        print(f"{label}: {len(peak_idx)} peak(s) above z = {threshold:.2f} "
              f"({detail}; ~{expected_false:.2g} false peaks "
              f"expected in this scan)"
              + "".join(f"\n  #{n}  t = {t:.5f}  z = {h:.1f}  post-flatten depth = {f['depth_ppm']:.0f} "
                        f"+- {f['depth_err_ppm']:.0f} ppm  (k = {f['k']:.4f}"
                        f"{', AT BOUND' if f['at_bound'] else ''})"
                        for n, t, h, f in zip(peak_numbers, peak_times, peak_heights, peak_fits)))

        results.append({
            "label": label,
            "time": time,
            "flux": processed["flux"],
            "flux_err": processed["flux_err"],
            "flattened_flux": flattened_flux,
            "flattened_err": flattened_err,
            "normalisation": processed["normalisation"],
            "time_trial": time_trial,
            "z": z,
            "S": S,
            "rho2": rho2,
            "threshold": threshold,
            "expected_false": expected_false,
            "mask_edges": mask_edges,
            "mask_widths": mask_widths,
            "z_threshold_override": z_threshold,
            "peak_indices": peak_idx,
            "peak_times": peak_times,
            "peak_heights": peak_heights,
            "peak_numbers": peak_numbers,
            "whitening_scale": scale,
            "window_days": window_days,
            "peak_k": [f["k"] for f in peak_fits],
            "peak_k_err_lo": [f["k_err_lo"] for f in peak_fits],
            "peak_k_err_hi": [f["k_err_hi"] for f in peak_fits],
            "peak_depth_ppm": [f["depth_ppm"] for f in peak_fits],
            "peak_depth_err_ppm": [f["depth_err_ppm"] for f in peak_fits],
            "peak_radius_1": [f["radius_1"] for f in peak_fits],
            "peak_fit_S_max": [f["S_max"] for f in peak_fits],
            "peak_fit_at_bound": [f["at_bound"] for f in peak_fits],
        })

    def cat(chunks):
        return np.concatenate(chunks) if len(chunks) else np.array([])

    global_data = {
        "labels": all_labels,
        "time": cat(all_time),
        "flux": cat(all_flux),
        "flux_err": cat(all_flux_err),
        "time_trial": cat(all_time_trial),
        "peak_times": np.array(all_peak_times),
        "peak_indices": np.array(all_peak_indices, dtype=int),
        "transit_time": cat(transit_time),
        "transit_flux": cat(transit_flux),
        "transit_flux_err": cat(transit_flux_err),
        "transit_time_segments": transit_time,
        "transit_flux_segments": transit_flux,
        "transit_flux_err_segments": transit_flux_err,
    }
    return results, global_data


# ======================================================
# Multiple-event statistic (Kepler-style MES on z)
# ======================================================

def _gather_events(results, stride_bins):
    """Interior epochs from all sectors, decimated so at most one epoch per
    transit lands in a phase bin (keeps the folded null exactly N(0,1)).

    Edge masking here is unconditional, independent of MASK_EDGES: the MES
    threshold rests on each folded contribution being N(0,1), and edge
    epochs (where the estimated baseline is unreliable) are exactly where
    that is only nominal."""
    times, cs, r2s = [], [], []
    for r in results:
        keep = interior_mask(r["rho2"])
        idx = np.flatnonzero(keep)[::stride_bins]
        times.append(r["time_trial"][idx])
        cs.append(r["z"][idx] * np.sqrt(r["rho2"][idx]))
        r2s.append(r["rho2"][idx])
    return np.concatenate(times), np.concatenate(cs), np.concatenate(r2s)


def multiple_event_search(results, width_days, min_period=10.0,
                          max_period=120.0, min_events=2):
    """Fold z over a period-epoch grid, as Kepler folds its SES into the MES.

    With c = z sqrt(rho2), MES(P, phase) = sum c / sqrt(sum rho2) is N(0,1)
    under the null and grows as sqrt(number of transits) for a real signal.
    Phase bins are half a width; the period grid keeps phase drift over the
    baseline below one bin.

    Returns a dict with the top 10 candidates (period, t0, mes, n_events;
    aliases within 1 per cent deduplicated), the grid size, and the
    threshold for ~1 expected false alarm over the whole search.
    """
    step = width_days / 2.0
    stride = max(1, int(round(step / np.median(
        [np.median(np.diff(r["time_trial"])) for r in results]))))
    t, c, r2 = _gather_events(results, stride)
    tref = t.min()
    baseline = t.max() - tref

    dlnp = step / baseline
    periods = np.exp(np.arange(np.log(min_period), np.log(max_period), dlnp))

    best = []
    for P in periods:
        nbins = max(1, int(np.ceil(P / step)))
        b = ((t - tref) % P / step).astype(np.int64)
        b[b >= nbins] = nbins - 1
        num = np.bincount(b, weights=c, minlength=nbins)
        den = np.bincount(b, weights=r2, minlength=nbins)
        cnt = np.bincount(b, minlength=nbins)
        ok = (den > 0) & (cnt >= min_events)
        if not ok.any():
            continue
        mes = np.where(ok, num / np.sqrt(np.where(den > 0, den, 1.0)), -np.inf)
        j = int(np.argmax(mes))
        best.append((float(mes[j]), float(P), tref + j * step, int(cnt[j])))

    best.sort(reverse=True)
    top = []
    for mes, P, t0, n in best:
        if all(abs(P - q["period"]) / q["period"] > 0.01 for q in top):
            top.append({"mes": mes, "period": P, "t0": t0, "n_events": n})
        if len(top) >= 10:
            break

    n_tests = sum(int(np.ceil(P / step)) for P in periods)
    return {"candidates": top, "n_periods": len(periods),
            "n_tests": n_tests,
            "threshold_1fa": float(stats.norm.isf(1.0 / n_tests)),
            "phase_bin_days": step}


# ======================================================
# Period search, ephemeris, folding
# ======================================================

def calculate_phase_dispersion(global_data, samples_per_peak=20,
                               minimum_period=10, maximum_period=200):
    """Phase-dispersion period search over the catalogued peaks."""
    periods, dispersion, chi_squared = bruce.template_match.phase_disperison(
        time_trial=global_data["time_trial"],
        peaks=global_data["peak_indices"],
        time=global_data["time"],
        flux=global_data["flux"],
        flux_err=global_data["flux_err"],
        samples_per_peak=samples_per_peak,
        minimum_period=minimum_period,
        maximum_period=maximum_period
    )
    return {"periods": periods, "dispersion": dispersion,
            "chi_squared": chi_squared}


def find_period_aliases(periods, dispersion, prominence=None, distance=10):
    """Dispersion minima ranked best-first; returns a list of dicts."""
    if prominence is None:
        prominence = 0.15 * np.ptp(dispersion)
    indices, properties = find_peaks(-dispersion, prominence=prominence,
                                     distance=distance)
    order = np.argsort(dispersion[indices])
    return [{"period": periods[indices[i]],
             "dispersion": dispersion[indices[i]],
             "prominence": properties["prominences"][i],
             "index": indices[i]} for i in order]


def linear_ephemeris(peak_times, reference_t0, initial_period):
    """Least-squares (period, t0) from peak times and a trial period."""
    orbit_number = np.round((peak_times - reference_t0) / initial_period)
    coeff = np.polyfit(orbit_number, peak_times, 1)
    return {"period": coeff[0], "t0": coeff[1], "orbit_number": orbit_number}


def phase_fold(global_data, period, t0):
    """Fold the concatenated light curve; phase in [-0.5, 0.5)."""
    phase = ((global_data["time"] - t0) / period) % 1
    phase = np.where(phase > 0.5, phase - 1, phase)
    order = np.argsort(phase)
    return {
        "phase": phase[order],
        "flux": global_data["flux"][order],
        "flux_err": global_data["flux_err"][order],
        "time": global_data["time"][order],
        "period": period,
        "t0": t0,
    }


def bin_phase_fold(folded_data, bin_width=0.01):
    """Bin a folded light curve; errors are standard errors on the mean."""
    phase = folded_data["phase"]
    flux = folded_data["flux"]
    bins = np.arange(phase.min(), phase.max() + bin_width, bin_width)

    bin_centres, bin_flux, bin_err = [], [], []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (phase >= left) & (phase < right)
        if not np.any(mask):
            continue
        bin_centres.append(np.mean(phase[mask]))
        bin_flux.append(np.mean(flux[mask]))
        bin_err.append(np.std(flux[mask], ddof=1) / np.sqrt(np.sum(mask))
                       if np.sum(mask) > 1 else np.nan)
    return {"phase": np.array(bin_centres), "flux": np.array(bin_flux),
            "flux_err": np.array(bin_err)}


def make_model(time, period, t_zero, radius_1, k):
    """Transit model flux at the given times."""
    return bruce.binarystar.lc(t=time, period=period, t_zero=t_zero,
                               radius_1=radius_1, k=k)


# ======================================================
# Plots
# ======================================================

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def plot_snr(results, filename=None):
    """z versus time for all scans, with numbered peaks and thresholds."""
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for result in results:
        ax.plot(result["time_trial"], result["z"], color="black", lw=0.8,
                zorder=1)
        for n, t, h in zip(result["peak_numbers"], result["peak_times"],
                           result["peak_heights"]):
            ax.scatter(t, h, s=5, c="red", marker="o", zorder=2)
            ax.text(t, h + 0.05 * max(abs(h), 1), str(n), ha="center",
                    va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5E6C8",
                              edgecolor="black", linewidth=0.8), zorder=5)
        ax.axhline(result["threshold"], color="grey", linestyle="--",
                   linewidth=0.6, zorder=3)

    ax.set_xlabel("BJD")
    ax.set_ylabel("Matched-filter S/N")
    ax.set_ylim(0, None)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_transit_grid(results, template, period, window=0.30, ncols=5,
                      filename=None):
    """One panel per detected peak: flattened flux, the model at that
    peak's own fitted depth (template width), peak z and depth."""
    panels = [(r, i) for r in results for i in range(len(r["peak_times"]))]
    if not panels:
        print("No transit detections")
        return

    rows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows=rows, ncols=ncols,
                             figsize=(2.0 * ncols, 2.0 * rows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, (result, i) in zip(axes, panels):
        number = result["peak_numbers"][i]
        peak_time = result["peak_times"][i]
        h = result["peak_heights"][i]
        time = result["time"]
        mask = (time > peak_time - window) & (time < peak_time + window)
        if "peak_k" in result:  # per-peak depth fit available
            k_i = result["peak_k"][i]
            r1_i = result["peak_radius_1"][i]
            k_err = np.nanmean([result["peak_k_err_lo"][i], result["peak_k_err_hi"][i]])
            label = (f"{number}  z={h:.0f}\nk = {k_i:.4f}"
                     + (f" ± {k_err:.4f}" if np.isfinite(k_err) else ""))
        else:  # older results: template depth
            k_i, r1_i = template["k"], template["radius_1"]
            label = f"{number}  z={h:.0f}"
        model = make_model(time[mask], period=period, t_zero=peak_time,
                           radius_1=r1_i, k=k_i)
        ax.errorbar(time[mask] - peak_time, result["flattened_flux"][mask],
                    yerr=result["flattened_err"][mask], fmt=".", ms=3,
                    color="black", ecolor="grey", elinewidth=0.5, capsize=0,
                    zorder=1)
        ax.plot(time[mask] - peak_time, model, "r", lw=1.5, zorder=3)
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5E6C8",
                          edgecolor="black", linewidth=0.8))
        ax.set_xlim(-window, window)

    for ax in axes[len(panels):]:
        fig.delaxes(ax)
    fig.supxlabel("Time from transit (days)")
    fig.supylabel("Flux")
    # bottom margin scales with the row count so the shared x label never
    # overlaps the tick labels (fixed 2-inch rows)
    plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.25 / rows + 0.05)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ======================================================
# Driver
# ======================================================

def main():
    print("zpipeline", VERSION)
    print("running from", __file__)
    set_plot_style()

    summary, datasets, labels, data_path = load_datasets(TIC_ID, use_ffi=False)
    if EXCLUDE_QLP:
        kept = [(d, l) for d, l in zip(datasets, labels) if "QLP" not in l]
        dropped = [l for l in labels if "QLP" in l]
        if dropped:
            print("Excluding QLP datasets:", ", ".join(dropped))
        datasets, labels = [d for d, _ in kept], [l for _, l in kept]
    reference_dataset = get_reference_dataset(datasets, labels, REFERENCE_SECTOR)

    template = z_optimise_template(
        reference_dataset,
        period=PERIOD,
        initial_radius_1=INITIAL_RADIUS_1,
        initial_k=INITIAL_K,
        epoch=EPOCH,
        step=STEP,
        window_days=WINDOW_DAYS
    )

    width = template_width(template["radius_1"], template["k"], PERIOD)
    print("\nOptimised template:")
    print("-------------------")
    print(f"flatten  = running median + boxcar, window {template['window_days']:.3f} d "
          f"= {template['window_days'] / width:.1f} x the optimised width")
    print(f"radius_1 = {template['radius_1']:.5f}  (width = {width * 24:.2f} h)")
    print(f"k        = {template['k']}  (scan placeholder; the depth is fitted per peak below)")
    print(f"t0       = {template['t0']:.5f}")

    results, global_data = z_run_template_matching(
        datasets, labels, template,
        period=PERIOD,
        step=STEP,
        window_days=WINDOW_DAYS,
        transit_window=TRANSIT_WINDOW,
        p_local=P_LOCAL,
        whiten=True,
        mask_edges=MASK_EDGES,
        mask_widths=EDGE_MASK_WIDTHS,
        z_threshold=Z_THRESHOLD,
        gap_days=SEGMENT_GAP_DAYS
    )

    with open(Z_RESULTS_FILE, "wb") as fh:
        pickle.dump({"template": template, "results": results,
                     "global_data": global_data}, fh)
    print(f"\nSaved {Z_RESULTS_FILE} "
          f"({len(global_data['peak_times'])} peaks over {len(results)} datasets)")

    tag = f"{TARGET_NAME}_{date.today():%Y%m%d}"
    plot_snr(results, filename=f"snr_{tag}.png")
    plot_transit_grid(results, template, period=PERIOD, window=0.30,
                      filename=f"grid_{tag}.png")
    print(f"Saved snr_{tag}.png and grid_{tag}.png")


if __name__ == "__main__":
    main()
