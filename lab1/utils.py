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

def autocovariance(timeseries, mode='same'):
    """
    Fill in this function! It should be able to handle shifts of varying lengths.
    """
    N = len(timeseries)

    sample_autocov = signal.correlate(timeseries, timeseries, mode=mode) / N
    if mode == 'same':
        shifts = np.arange(-N//2, N//2)
    elif mode == 'full':
        shifts = np.arange(-N, N)
    else:
        shifts = None

    return sample_autocov, shifts


def periodogram(timeseries, fs = 1):
    
    psd = fft.fft(timeseries)
    freqs = np.linspace(0., fs, num=len(psd))

    return np.abs(psd)**2 / len(timeseries), freqs

def periodogram_tapered(timeseries, taper, fs=1):
    
    sig = timeseries * taper
    return periodogram(sig, fs=fs)


def multitaper_periodogram(timeseries, nw=4, ntapers = 7, fs = 1):
    tapers = signal.windows.dpss(len(timeseries), nw, Kmax=ntapers) * np.sqrt(len(timeseries))

    # Loop through tapers to compute different spectral estimates
    powers = []
    for taper in tapers:
        powers.append(periodogram_tapered(timeseries, taper)[0])

    _, freqs = periodogram(timeseries, fs=fs)
    S_yy_est = np.asanyarray(powers)
    power = S_yy_est.mean(axis=0)
    return power, freqs

def multitaper_spectrogram(timeseries, window_size = 1024, nw = 4, ntapers = 4, fs=1):
    N = len(timeseries)

    nwindows = N // window_size
    spectrogram = np.zeros((window_size, nwindows))

    for i in range(nwindows):
        spectrogram[:,i] = multitaper_periodogram(timeseries[i*window_size:(i+1)*window_size], nw=nw, ntapers=ntapers, fs=fs)[0]

    freqs = np.linspace(0, fs, num=window_size)
    times = np.linspace(0, num=nwindows, stop=nwindows*(window_size / fs))

    return spectrogram, freqs, times