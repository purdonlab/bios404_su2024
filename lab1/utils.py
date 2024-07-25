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


# Copy the funtion definition `compute_periodogran` to here, and remove
# these commented line.


# Copy the funtion definition `compute_tapered_periodogran` to here, and remove
# these commented line.


# Copy the funtion definition `compute_multitaper_spectrum` to here, and remove
# these commented line.
