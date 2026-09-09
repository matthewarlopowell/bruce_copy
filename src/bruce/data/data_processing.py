import warnings
import bruce_c, numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.interpolate import UnivariateSpline

###############################################################
#                  bruce_c wrappers                           #
###############################################################
MEDIAN_FILTER_MAX_POINTS = 1024      # bruce_c MAX_WINDOW_SIZE: points kept per median window
NORMALISATION_WINDOW_WIDTHS = 5.0    # default normalisation window = five transit widths ...
NORMALISATION_WINDOW_FLOOR = 0.2     # ... but never shorter than 0.2 d


def median_filter(time, flux, bin_size=0.5/24/3):
    """Running median of ``flux`` over a window of ``bin_size`` (same units
    as ``time``; total length, symmetric about each point).  The C kernel
    keeps at most MEDIAN_FILTER_MAX_POINTS points per window and silently
    drops the rest from the right-hand side, so a warning is issued when a
    window would hold more than that (bin the data or shorten the window)."""
    time = np.asarray(time, dtype=np.float64)
    if time.size > 1:
        n_points = float(bin_size) / float(np.nanmedian(np.diff(time)))
        if n_points > MEDIAN_FILTER_MAX_POINTS:
            warnings.warn(
                "median_filter: a window of %g spans %.0f cadences but the C "
                "kernel uses at most %d points per window (truncated "
                "one-sidedly); bin the data or shorten the window."
                % (bin_size, n_points, MEDIAN_FILTER_MAX_POINTS), stacklevel=2)
    return bruce_c.median_filter(time, flux, bin_size)


def convolve_1d(time, flux, bin_size=0.5/24/3) : return bruce_c.convolve_1d(time, flux, bin_size)
def bin_data(time, flux, bin_size=0.5/24/3) : return bruce_c.bin_data(time, flux, bin_size)
def check_proximity_of_timestamps(time_trial, time, width) : return bruce_c.check_proximity_of_timestamps(time_trial, time, width)


def normalisation_window(width, window_widths=NORMALISATION_WINDOW_WIDTHS,
                         floor=NORMALISATION_WINDOW_FLOOR):
    """Default length (days) of the normalisation window for a transit of
    duration ``width`` (days): max(floor, window_widths * width).

    Five transit widths keep the in-transit points well under half of the
    median window (the median is robust while they are a minority) and is
    the default of the search pipeline and of the paper's simulations.
    Shorten it for stars whose variability timescale is shorter, but keep
    it at three widths or more.
    """
    return max(float(floor), float(window_widths) * float(width))


def normalisation_model(time, flux, width=None, window=None,
                        window_widths=NORMALISATION_WINDOW_WIDTHS,
                        floor=NORMALISATION_WINDOW_FLOOR):
    """Out-of-transit baseline w_n: a running median of length ``window``
    smoothed by a boxcar of the same length (median_filter then
    convolve_1d).

    ``window`` (days) defaults to normalisation_window(width); pass
    ``window`` explicitly to fix the length.  The median is robust to the
    sparse in-transit points, so it absorbs several times less of the
    transit and about half as much noise variance as a moving average of
    the same length.  Warns if the window is under twice the transit width.
    """
    if window is None:
        if width is None:
            raise ValueError("normalisation_model needs `width` (transit "
                             "duration, days) or an explicit `window` (days)")
        window = normalisation_window(width, window_widths, floor)
    window = float(window)
    if width is not None and window < 2.0 * float(width):
        warnings.warn(
            "normalisation window %.3f d is under twice the transit width "
            "(%.1f h): the median will follow the transit; use >= 3 widths."
            % (window, float(width) * 24.0), stacklevel=2)
    t64 = np.ascontiguousarray(time, dtype=np.float64)
    f64 = np.ascontiguousarray(flux, dtype=np.float64)
    track = median_filter(t64, f64, window)
    return convolve_1d(t64, track, window)



###############################################################
#                  Other functions                            #
###############################################################
def find_nights_from_data(x, dx_lim):
    '''
    Split array buy time gaps. Can be used to get individual nights in datasets.
    '''
    dx = np.gradient(x) 
    dx_thresh = np.sort(np.where(dx >= dx_lim)[0] +1)

    # Now check for consecutive integers and delte them
    delete_idxs = []
    i = 0 
    while i < (dx_thresh.shape[0]-1):
        if (dx_thresh[i]+1) == dx_thresh[i+1] :
            delete_idxs.append(i+1)
            i +=2
        else : i +=1
    dx_thresh = np.delete(dx_thresh, delete_idxs)

    # create the idx to split
    idx = np.arange(x.shape[0])
    d =  np.split(idx, dx_thresh)
    return [i for i in d if i.shape[0]>0]



def mags_to_flux(mags, mags_err):
    """
    Take 2 arrays, light curve and errors
    and convert them from differential magnitudes
    back to relative fluxes
    Rolling back these equations:
        mags = - 2.5 * log10(flux)
        mag_err = (2.5/log(10))*(flux_err/flux)
    """
    flux = 10.**(mags / -2.5)
    flux_err = (mags_err/(2.5/np.log(10))) * flux
    return flux, flux_err

def flux_to_mags(flux, flux_err):
    """
    Take 2 arrays, light curve and errors
    and convert them from differential magnitudes
    back to relative fluxes
    Applying these equations:
        mags = - 2.5 * log10(flux)
        mag_err = (2.5/log(10))*(flux_err/flux)
    """
    mags = -2.5*np.log10(flux)
    mags_err = (2.5/np.log(10))*(flux_err/flux)
    return mags, mags_err

def phase_times(times, epoch, period, phase_offset=0.0):
    """
    Take a list of times, an epoch, a period and
    convert the times into phase space.
    An optional offset can be supplied to shift phase 0
    """
    return (((times - epoch)/period)+phase_offset)%1 - phase_offset


def flatten_data_with_function(time, flux, method = 'savgol', max_median=0, npoly = 3, Nmaxfilter=11, Nmedianfilter=19, splinesmooth=100, SG_window_length=10, SG_polyorder=3, SG_deriv=0, SG_delta=1., SG_iter=1, SG_sigma=3):
    # Do a maximum/median filter, if needed. 
    if max_median == 1 : flux = median_filter(maximum_filter(flux, Nmaxfilter), Nmedianfilter)
    elif max_median == 2 : flux = maximum_filter(median_filter(flux, Nmedianfilter), Nmaxfilter)


    # Now enter the flatten methods
    if method=='poly1d': 
        return np.poly1d(np.polyfit(time, flux, npoly))(time)
    elif method=='spline':
        spl = UnivariateSpline(time, flux)
        spl.set_smoothing_factor(30000)
        return spl(time)
    elif method=='savgol':
        if SG_window_length > len(time) : 
            SG_window_length = int(0.75*len(time))
            if ((SG_window_length % 2) ==0) : SG_window_length -= 1 # make sure its an odd number
        if SG_iter==1 : return savgol_filter(flux, window_length=SG_window_length, polyorder=SG_polyorder, deriv=SG_deriv, delta=SG_delta)
        else:
            mask = np.ones(len(flux), dtype = bool)
            flux_ = np.copy(flux) 
            trend = np.copy(flux) 

            for i in range(SG_iter):
                trend[mask] = savgol_filter(flux[mask], window_length=SG_window_length, polyorder=SG_polyorder, deriv=SG_deriv, delta=SG_delta)
                std = np.std(flux_[mask] - trend[mask])
                mask[np.abs(flux - trend) > SG_sigma*std] = 0
                #mask[sigma_clip(flux - trend, sigma_lower=SG_sigma, sigma_upper=0.5*SG_sigma).mask] = 0

            return np.interp(time, time[mask], trend[mask])
