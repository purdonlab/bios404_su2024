# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Tom Stone <email:tomstone@stanford.edu>
# License: BSD (3-clause) 
import numpy as np
from scipy import linalg, signal, fft


def compute_theoritical_ar_acov(phi, p):
    """scalar case of the following
    http://dx.doi.org/10.1016/j.spl.2016.12.015"""
    phi = np.append(phi, np.zeros(p - phi.shape[-1] + 1))
    c = np.zeros(p+1)
    r = np.zeros(2*p+1)
    c[0] = phi[-1]
    r[:phi.shape[-1]] = phi[::-1]
    a = linalg.toeplitz(c, r)
    a_bar = np.zeros((p+1, p+1))
    a_bar[:, 1:] += a[:, :p][:, ::-1]
    a_bar += a[:, p:]
    a_bar_inv = linalg.pinv(a_bar)
    acov = np.zeros(2*p+1)
    acov[p:] = a_bar_inv[:, 0]
    acov[:p] = a_bar_inv[1:, 0][::-1]
    times = np.arange(-p, p+1)
    return acov, times


# Copy the function definition of`compute_autocovaraince()` to here, and
# remove these commented line.
# You shall keep in mind that two function definitions should be spaced by
# two blank lines.
def compute_autocovaraince(x, max_lag=512):
    """Computes autocovarinane upto a given lag.
    
    Parameters:
        x:
            the signal
        max_lag: 
            maximum lag to consider for autocovaraince sequenc
    Returns:
        acov:
            the sample autocovariance, normalized by the number of samples.
        time_indices:
         integer shifts, i.e the x-axis for the autocovariance plot.
    """
    assert max_lag > 0, f"max_lag needs to be >0, received {max_lag}"
    n = x.shape[-1] # length of time series
    assert max_lag < n, f"max_lag needs to be < signal length ({n}) , received {max_lag}"
    # Compute the mean, `mu`
    mu = x.sum(axis=-1) / n

    # Remove the mean from the sequence
    x = x - mu

    # Use `signal.correlate` function to compute the acov sequence
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
    # Use mode='full' which returns the full result, i.e. lag -n+1 to lag n-1
    acov = signal.correlate(x, x, mode='full', method='fft') / n
    # Create an index array corrsponding to the correlation values using `numpy.arange``
    indices = np.arange(-n+1, n)

    # Create a boolean selection to extract indices within range[-max_lag, max_lag].
    # Use `numpy.logical_and()`
    selection = np.logical_and(indices < max_lag+1, indices > -max_lag-1)
    return acov[selection], indices[selection]


# Copy the funtion definition `compute_periodogran` to here, and remove
# these commented line.
def compute_periodogram(x):
    """Compute periodogram using fft directly on the signal.
    
    Parameters:
        x:
            the signal
    Returns:
        S_xx:
            periodogram.
        freqs:
         associated frequency (normalized) points.
    """
    n = x.shape[-1]
    # Compute the mean
    mu = x.sum() / n
    # remove the mean from the signal
    x = x - mu
    
    # Now use the `fft.fft()` funtion to perform the computation.
    S_xx = np.abs(fft.fft(x)) ** 2 / n
    freqs = np.linspace(0., 1, num=n)
    return S_xx, freqs


# Copy the funtion definition `compute_periodogran` to here, and remove
# these commented line.
def compute_tapered_periodogram(x, window_type):    
    """Compute periodogram using after applying given taper

    Parameters:
        x:
            the signal
        window_type: 
            the taper to apply before computing periodogram
    Returns:
        S_xx:
            periodogram.
        freqs:
         associated frequency (normalized) points.
    """
    n = x.shape[-1]
    # generate the taper
    taper = signal.get_window(window_type, n)

    # taper the signal
    tapered_x = x * taper

    # Compute the periodogram using your written `compute_periodogram` function
    # This is 'reuse' of code, and highly encouraged!!!
    St_xx, freqs = compute_periodogram(tapered_x)
    return St_xx, freqs


def multitaper_periodogram(timeseries, nw=4, ntapers=4, fs=1):
    """
    Fill in this function!
    arguments:
        timeseries - timeseries you are computing the multitaper periodogram of
        nw - time-half-bandwidth product, defaults to 4
        ntapers - number of DPSS tapers to use, defaults to 4
        fs - sampling frequency, defaults to 1
    returns:
        multitaper_psd - multitaper periodogram
        freqs - same as before
    """
    N = len(timeseries)
    tapers = signal.windows.dpss(N, nw, Kmax=ntapers) * np.sqrt(N)

    # Loop through tapers to compute different spectral estimates
    S_yy_est = np.array((ntapers, N))

    for (i,taper) in enumerate(tapers):
        S_yy_est[i,:] = periodogram_tapered(timeseries, taper)[0]

    multitaper_psd = S_yy_est.mean(axis=0)

    freqs = np.linspace(0, fs, num=N)

    return multitaper_psd, freqs


def multitaper_spectrogram(timeseries, window_size = 1024, nw = 4, ntapers = 4, fs=1):
    """
    Fill in this function!
    arguments:
        timeseries - timeseries of which to compute the multitaper spectrogram
        window_size - length of segments to break the timeseries into, defaults to 1024
        nw - time-half-bandwidth product, defaults to 4
        ntapers - number of DPSS tapers to use, defaults to 4
        fs - sampling frequency of the data, defaults to 1
    returns:
        spectrogram - 2D numpy array of spectrogram, first axis is frequency axis, second axis is time axis.
        freqs - same as before
        times - the starting time of all the windows, in seconds.
    """
    N = len(timeseries)

    nwindows = N // window_size
    spectrogram = np.zeros((window_size, nwindows))

    for i in range(nwindows):
        spectrogram[:,i] = multitaper_periodogram(timeseries[i*window_size:(i+1)*window_size], nw=nw, ntapers=ntapers, fs=fs)[0]

    freqs = np.linspace(0, fs, num=window_size)
    times = np.linspace(0, nwindows*(window_size / fs), num=nwindows)

    return spectrogram, freqs, times