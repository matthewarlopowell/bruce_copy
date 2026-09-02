import warnings

import numpy as np
from bruce.binarystar import transit_width
from bruce.data import check_proximity_of_timestamps
import bruce_c
import scipy.stats as stats
from scipy.integrate import quad
from scipy.optimize import brentq


def template_match_lightcurve(time, flux, flux_err, normalisation_model, period = 1.,
        radius_1=0.2, k = 0.2, incl=np.pi/2,
        e=0., w = np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = -2,
        accurate_tp=1,
		jitter=0., offset=0,
		time_step=None, time_trial=None):
	# Data
	time, flux, flux_err = time.astype(np.float64), flux.astype(np.float64), flux_err.astype(np.float64)

	if isinstance(radius_1, np.ndarray):
		width = 0.5 # Fix this
		if time_trial is None:
			if time_step is None : time_step = width / 20.
			time_trial = np.arange(np.min(time) - width/2., np.max(time)+width/2., time_step)
			time_trial_mask = check_proximity_of_timestamps(time_trial, time, width)
			time_trial = time_trial[time_trial_mask]
		# Call
		DeltaL = bruce_c.template_match_batch_reduce(time_trial,
			time, flux, flux_err, normalisation_model,
			period,
			radius_1, k, incl,
			e, w,
			c, alpha,
			cadence, noversample,
			light_3,
			ld_law,
			accurate_tp,
			jitter, offset)
	else:
		# Get the width
		width = transit_width(radius_1, k, np.cos(incl)/radius_1, period=period)

		# Check the time steps
		if time_trial is None:
			if time_step is None : time_step = width / 20.
			time_trial = np.arange(np.min(time) - width/2., np.max(time)+width/2., time_step)
			time_trial_mask = check_proximity_of_timestamps(time_trial, time, width)
			time_trial = time_trial[time_trial_mask]
		# Call
		DeltaL = bruce_c.template_match_reduce(time_trial,
			time, flux, flux_err, normalisation_model,
			width,
			period,
			radius_1, k, incl,
			e, w,
			c, alpha,
			cadence, noversample,
			light_3,
			ld_law,
			accurate_tp,
			jitter, offset)

	return time_trial, DeltaL


###############################################################
#     Calibrated detection statistic and FAP thresholds       #
###############################################################
#
# template_match_lightcurve returns the *fixed-template* statistic
#
#     S(t_j) = 2 dlnL = sum_n [ (F_n - w_n)^2 - (F_n - w_n m_n)^2 ] / sigma_n^2
#
# with nothing fitted per trial epoch, so Wilks' theorem does not apply and S
# is not chi-squared under the null for any df.  Writing d_n = w_n (1 - m_n)
# and rho^2(t_j) = sum_n d_n^2 / sigma_n^2 (the template norm), the exact
# white-noise null is Gaussian and epoch-dependent:
#
#     S ~ N(-rho^2, (2 rho)^2)
#
# The calibrated detection variable is the amplitude-profiled (GLRT)
# statistic, identical to the whitened matched-filter S/N:
#
#     z = (S + rho^2) / (2 rho),    z ~ N(0, 1) under the null.
#
# Thresholds are quoted on z (constant per FAP) or, equivalently, on S with
# *per-epoch* heights T_j = 2 rho_j z_p - rho_j^2.


def template_rho2(time, flux_err, normalisation_model, time_trial=None, **kwargs):
    """Per-epoch template norm rho^2 via the self-match identity.

    Matching the normalisation model against itself leaves zero residual for
    the null term, so S_self = -rho^2 exactly.  rho^2 = sum w_n^2 (1-m_n)^2
    / sigma_n^2 depends on the template, the stated errors, the time
    sampling, AND the normalisation model -- not on the flux directly -- so
    it can only be reused across scans that share all four.  A flux-derived
    normalisation model (e.g. a median filter) differs per light curve, in
    which case rho^2 must be recomputed (though for normalised flux with
    w ~ 1 the difference is at the 1e-3 level).
    """
    normalisation_model = np.asarray(normalisation_model, dtype=np.float64)
    time_trial, s_self = template_match_lightcurve(
        time, normalisation_model, flux_err, normalisation_model,
        time_trial=time_trial, **kwargs)
    return time_trial, -np.asarray(s_self)


def template_match_snr(time, flux, flux_err, normalisation_model,
                       whiten=True, rho2=None, **kwargs):
    """Whitened matched-filter S/N per trial epoch; the null is N(0, 1).

    Returns ``(time_trial, z, S, rho2)`` where ``S`` is the raw 2dlnL from
    ``template_match_lightcurve`` and ``z = (S + rho2) / (2 sqrt(rho2))``.

    ``whiten=True`` divides z by its robust (MAD) scatter: an estimated
    normalisation model (e.g. a median filter) absorbs part of the noise and
    leaves the raw z slightly narrower than N(0,1); transits are sparse so
    the MAD is insensitive to them.  The scatter is estimated from the
    interior trial epochs only (rho2 above half its median): edge-affected
    epochs carry a deflated z and would bias the scale low.  The scale needs
    several independent samples and the z process decorrelates over about
    one template width, so scans spanning fewer than 3 widths skip the
    whitening with a warning (the raw z is returned).  Pass a precomputed
    ``rho2`` (from ``template_rho2``) to skip the self-match pass.
    """
    if isinstance(kwargs.get("radius_1"), np.ndarray):
        raise ValueError("template_match_snr supports the single-template "
                         "path only (scalar radius_1, k, incl)")
    kwargs = dict(kwargs)
    time_trial_in = kwargs.pop("time_trial", None)
    time_trial, S = template_match_lightcurve(
        time, flux, flux_err, normalisation_model,
        time_trial=time_trial_in, **kwargs)
    if rho2 is None:
        _, rho2 = template_rho2(time, flux_err, normalisation_model,
                                time_trial=time_trial, **kwargs)
    S = np.asarray(S, dtype=float)
    rho2 = np.asarray(rho2, dtype=float)
    safe = np.maximum(rho2, 1e-12)
    good = rho2 > 1e-9
    z = np.where(good, (S + rho2) / (2.0 * np.sqrt(safe)), 0.0)
    if whiten:
        # Neighbouring grid epochs are strongly correlated (step ~ W/20), so
        # a pool that is large in epochs can still hold ~no independent
        # samples: on a scan of ~1 template width the MAD measures the
        # within-correlation-length wiggle and inflates z arbitrarily.
        radius_1_ = float(kwargs.get("radius_1", 0.2))
        b_ = abs(np.cos(float(kwargs.get("incl", np.pi / 2)))) / max(radius_1_, 1e-9)
        try:
            width_ = float(transit_width(radius_1_, float(kwargs.get("k", 0.2)),
                                         b_, period=float(kwargs.get("period", 1.))))
        except Exception:
            width_ = 0.0
        span_ = float(time_trial.max() - time_trial.min()) if time_trial.size else 0.0
        if np.isfinite(width_) and width_ > 0.0 and span_ < 3.0 * width_:
            warnings.warn(
                "scan spans only %.1f template widths; the per-scan MAD "
                "scale needs several independent samples (z decorrelates "
                "over ~1 width), so whitening is skipped and the raw z is "
                "returned" % (span_ / width_), stacklevel=2)
        else:
            valid = good & np.isfinite(z)
            if good.any():
                valid &= rho2 > 0.5 * np.median(rho2[good])
            pool = z[valid]
            if pool.size < 10:
                pool = z[np.isfinite(z)]
            scale = 1.4826 * np.median(np.abs(pool - np.median(pool)))
            if np.isfinite(scale) and scale > 0.0:
                z = z / scale
    return time_trial, z, S, rho2


def get_snr_height_from_fap(p_value=(1e-2, 1e-3, 1e-4), n_independent=1):
    """One-sided N(0,1) heights on the matched-filter S/N z.

    With the default ``n_independent=1`` the height is LOCAL (per trial
    epoch) -- threshold there and the expected number of false peaks in a
    scan is ~ N_eff * p.  For a whole-scan (global) false-alarm
    probability instead, pass
    ``n_independent = effective_independent_trials(time_trial, width)``,
    which is N_eff ~ 5 T/W -- a scan carries about five independent tests
    per transit width (Monte Carlo measured, 4.4-5.3; see that function's
    docstring).  The correction is Sidak,
    ``p_local = 1 - (1 - p)**(1/n_independent)`` (~ p/N_eff), the same
    arithmetic as Kepler's 7.1-sigma threshold for ~2e12 tests.

    Example -- threshold for a 1e-4 chance of ANY false peak in a 27 d
    scan with a 6 h template (N_eff ~ 540):

        n = effective_independent_trials(time_trial, width)
        _, (z_p,) = get_snr_height_from_fap([1e-4], n_independent=n)
        # z_p ~ 5.1, versus 3.72 for the local 1e-4

    ``snr_threshold_global_upcross`` gives a sharper analytic threshold in
    the far tail; Monte Carlo remains the gold standard for
    publication-grade numbers.
    """
    p = np.atleast_1d(np.asarray(p_value, dtype=float))
    p_local = 1.0 - (1.0 - p) ** (1.0 / max(float(n_independent), 1.0))
    return p, stats.norm.isf(p_local)


def effective_independent_trials(time_trial, width, decorrelation_factor=5.0):
    """Effective number of independent template placements in a scan.

    Neighbouring trial epochs share in-transit points, so the scan carries
    roughly one independent test per correlation length of the z process.
    Monte Carlo on the W/20 grid (2,000 white-noise scans, 27 d at 10-min
    cadence, 6 h template; corroborated across scan durations of 6-54 d
    and template widths of 2-24 h) measures 4.4-5.3 independent tests per
    width at moderate significance, hence the default factor of 5.  This is a
    cheap fixed-N Sidak estimate and degrades in the far tail, where the
    effective N keeps growing with the threshold height; prefer
    ``snr_threshold_global_upcross`` for whole-scan thresholds.
    """
    time_trial = np.asarray(time_trial, dtype=float)
    if time_trial.size < 2:
        return 1.0
    step = float(np.median(np.diff(time_trial)))
    return max(1.0, decorrelation_factor * time_trial.size * step / float(width))


def lag1_correlation(z):
    """Lag-1 autocorrelation of the z series (dominated by the null when
    transits are sparse); used by ``snr_threshold_global_upcross``."""
    z = np.asarray(z, dtype=float)
    good = np.isfinite(z[:-1]) & np.isfinite(z[1:])
    if good.sum() < 10:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(z[:-1][good], z[1:][good])[0, 1])
    return r if np.isfinite(r) else 0.0


def _bvn_survival_equal(u, r):
    """P(Z1 > u, Z2 > u) for a standard bivariate normal with correlation r.

    Uses Plackett's identity (d Phi2 / d rho is the bivariate density, which
    at the equal point (u, u) is exp(-u^2/(1+rho)) / (2 pi sqrt(1-rho^2))),
    integrating from independence -- numerically stable in the far tail.
    """
    def density(rho):
        return np.exp(-u * u / (1.0 + rho)) / (2.0 * np.pi * np.sqrt(1.0 - rho * rho))
    integral, _ = quad(density, 0.0, r, limit=200)
    return stats.norm.sf(u) ** 2 + integral


def upcrossing_probability(u, r1):
    """P(z_i <= u < z_{i+1}) for consecutive grid points with correlation r1."""
    return stats.norm.sf(u) - _bvn_survival_equal(u, r1)


def snr_threshold_global_upcross(fap, n_grid, r1):
    """Whole-scan threshold on z from the discrete upcrossing rate.

    Expected entries into the exceedance region:
    E(u) = Q(u) + (n_grid - 1) P(z_i <= u < z_{i+1}); Poisson clumping gives
    P(max z > u) ~ 1 - exp(-E(u)).  Adapts to the actual grid spacing and
    the measured lag-1 correlation of the z series (cf. Rice/Davies bounds
    and Baluev 2008, MNRAS 385, 1279 for periodograms).
    """
    fap = float(fap)
    r1 = min(max(float(r1), 0.0), 0.999999)

    def objective(u):
        expected = stats.norm.sf(u) + (n_grid - 1) * upcrossing_probability(u, r1)
        # -expm1(-E) = 1 - exp(-E) without cancellation: 1.0 - np.exp(-E)
        # rounds to 0 for E < ~1e-16 and the far tail would be lost.
        return -np.expm1(-expected) - fap

    return brentq(objective, 0.5, 12.0, xtol=1e-6)


def get_delta_loglike_height_from_fap(p_value=(0.01, 0.001, 0.0001), df=None,
                                      rho2=None, n_independent=1):
    """Detection heights on BRUCE's 2dlnL statistic for the requested FAPs.

    Because the null of S is N(-rho^2, (2 rho)^2), a constant height cannot
    hold a fixed false-alarm probability; the valid height is per-epoch:

        height_j = 2 sqrt(rho2_j) * z_p - rho2_j,   z_p = Phi^-1(1 - p_local)

    Pass ``rho2`` from ``template_rho2`` (or ``template_match_snr``); the
    returned ``heights`` has shape ``(len(p_value), len(rho2))`` and each row
    can be given directly to ``scipy.signal.find_peaks(S, height=row)``.
    ``S > height_j`` is algebraically identical to ``z > z_p``.
    ``n_independent`` applies a Sidak correction for whole-scan FAPs.
    Epochs with ``rho2 <= 0`` (no data within the transit span -- possible
    when a caller supplies an explicit ``time_trial`` extending past the
    data or across wide gaps) get ``height = +inf``: S = 0 exactly there,
    so a finite height of 0 would let find_peaks flag pure noise as a
    detection.

    The legacy chi-squared quantile (``df`` given, no ``rho2``) is retained
    for backwards compatibility only: it is statistically invalid for this
    statistic (nothing is fitted per epoch, so Wilks' theorem does not
    apply) and its realised FAP is uncontrolled and template-depth
    dependent.  It emits a DeprecationWarning.
    """
    if rho2 is None:
        if df is None:
            df = 6
        warnings.warn(
            "chi2(df) heights are statistically invalid for the "
            "fixed-template 2dlnL (its null is N(-rho^2, 4 rho^2), not "
            "chi-squared); pass rho2 from template_rho2/template_match_snr "
            "for calibrated per-epoch heights, or threshold the z statistic "
            "with get_snr_height_from_fap.", DeprecationWarning, stacklevel=2)
        return p_value, stats.chi2.ppf(1 - np.array(p_value), df)
    p, z_p = get_snr_height_from_fap(p_value, n_independent)
    rho2 = np.atleast_1d(np.asarray(rho2, dtype=float))
    rho = np.sqrt(np.maximum(rho2, 0.0))
    heights = 2.0 * rho[None, :] * z_p[:, None] - rho2[None, :]
    heights[:, rho2 <= 0.0] = np.inf
    return p, heights


def phase_disperison(time_trial, peaks, time, flux, flux_err,
					 	periods=None,
						samples_per_peak=5,
						nyquist_factor=5,
						minimum_period=None,
						maximum_period=None):
    if periods is None : periods = autoperiod(time_trial, samples_per_peak=samples_per_peak, nyquist_factor=nyquist_factor,minimum_period=minimum_period,maximum_period=maximum_period)
    dispersion ,chi_squared = bruce_c.phase_dispersion(time_trial, np.array(peaks, dtype=np.int32),  periods, time, flux, flux_err)
    return periods, dispersion ,chi_squared


def autoperiod( x,
    samples_per_peak=5,
    nyquist_factor=5,
    minimum_period=None,
    maximum_period=None,
    return_freq_limits=False,
):
    """Determine a suitable frequency grid for data.

    Note that this assumes the peak width is driven by the observational
    baseline, which is generally a good assumption when the baseline is
    much larger than the oscillation period.
    If you are searching for periods longer than the baseline of your
    observations, this may not perform well.

    Even with a large baseline, be aware that the maximum frequency
    returned is based on the concept of "average Nyquist frequency", which
    may not be useful for irregularly-sampled data. The maximum frequency
    can be adjusted via the nyquist_factor argument, or through the
    maximum_frequency argument.

    Parameters
    ----------
    samples_per_peak : float, optional
        The approximate number of desired samples across the typical peak
    nyquist_factor : float, optional
        The multiple of the average nyquist frequency used to choose the
        maximum frequency if maximum_frequency is not provided.
    minimum_frequency : float, optional
        If specified, then use this minimum frequency rather than one
        chosen based on the size of the baseline.
    maximum_frequency : float, optional
        If specified, then use this maximum frequency rather than one
        chosen based on the average nyquist frequency.
    return_freq_limits : bool, optional
        if True, return only the frequency limits rather than the full
        frequency grid.

    Returns
    -------
    frequency : ndarray or `~astropy.units.Quantity` ['frequency']
        The heuristically-determined optimal frequency bin
    """
    baseline = x.max() - x.min()
    n_samples = x.size

    df = 1.0 / baseline / samples_per_peak

    if maximum_period is None : minimum_frequency = 0.5 * df
    else : minimum_frequency = 1 / maximum_period

    if minimum_period is None:
        avg_nyquist = 0.5 * n_samples / baseline
        maximum_frequency = nyquist_factor * avg_nyquist
    else : maximum_frequency = 1 / minimum_period

    Nf = 1 + int(np.round((maximum_frequency - minimum_frequency) / df))

    if return_freq_limits:
        return minimum_frequency, minimum_frequency + df * (Nf - 1)
    else:
        return 1/(minimum_frequency + df * np.arange(Nf))
