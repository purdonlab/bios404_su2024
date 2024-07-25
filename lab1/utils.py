# Author: Proloy Das <email:proloyd94@gmail.com>
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


# Copy the funtion definition `compute_tapered_periodogran` to here, and remove
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


# Copy the funtion definition `compute_multitaper_spectrum` to here, and remove
# these commented line.
def compute_multitaper_spectrum(x, NW, Kmax):
    """Compute multitaper spectrum with Kmax tapers of NW 
    time-bandwidth product

    Parameters:
        x:
            the signal
        NW: 
            time-bandwidth product
        Kmax:
            Number of tapers to use
    Returns:
        S_xx:
            periodogram.
        freqs:
         associated frequency (normalized) points.
    """
    n = x.shape[-1]
    # generate the tapers, use norm=2 keyword
    dpss_tapers = signal.windows.dpss(n, NW, Kmax=Kmax, norm=2)
    # These tapers are normalized by window length, so you need to take care of them.

    # Cycle through the tapers in a for loop, and get the tapered estimate
    # Store the generate estimates in a list.
    St_xxs = []
    for taper in dpss_tapers:
        # taper the signal
        tapered_x = x * taper

        # Compute the periodogram using your written `compute_periodogram` function
        # This is 'reuse' of code, and highly encouraged!!!
        St_xx, freqs = compute_periodogram(tapered_x)
        St_xxs.append(St_xx * n)
    # We will stack the rows in the list vertically to create a numpy array
    St_xxs = np.vstack(St_xxs)
    # Then take its mean in first (0th) dimension to get the multitaper estimate
    Smtm_xx = St_xxs.mean(0)
    return Smtm_xx, freqs
