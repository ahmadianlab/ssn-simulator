import numpy as np
#import jax.numpy as np

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, inds=None, verbose=False, silent=False, Tfrac_CV=0):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot
    verbose: if True print convergence criteria even if passed (function always prints out a warning if it doesn't converge).
    Tfrac_var: if not zero, maximal temporal CV (coeff. of variation) of state vector components, over the final
               Tfrac_CV fraction of Euler timesteps, is calculated and printed out.
               
    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

    if PLOT:
        if inds is None:
            x_dim = x_initial.size
            inds = [int(x_dim/4), int(3*x_dim/4)]
        xplot = x_initial.flatten()[inds][:,None]

    Nmax = int(np.round(Tmax/dt))
    Nmin = int(np.round(Tmin/dt)) if Tmax > Tmin else int(Nmax/2)
    xvec = x_initial
    CONVG = False

    if Tfrac_CV > 0:
        xmean = np.zeros_like(xvec)
        xsqmean = np.zeros_like(xvec)
        Nsamp = 0

    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        if PLOT:
            xplot = np.hstack((xplot, xvec.flatten()[inds][:,None]))
        
        if Tfrac_CV > 0 and n >= (1-Tfrac_CV) * Nmax:
            xmean = xmean + xvec
            xsqmean = xsqmean + xvec**2
            Nsamp = Nsamp + 1

        if n > Nmin:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol:
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG and not silent: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))

        if Tfrac_CV > 0:
            xmean = xmean/Nsamp
            xvec_SD = np.sqrt(xsqmean/Nsamp - xmean**2)
            # CV = xvec_SD / xmean
            # CVmax = CV.max()
            CVmax = xvec_SD.max() / xmean.max()
            print(f"max(SD)/max(mean) of state vector in the final {Tfrac_CV:.2} fraction of Euler steps was {CVmax:.5}")

        #mybeep(.2,350)
        #beep

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(244459)
        plt.plot(np.arange(n+2)*dt, xplot.T, 'o-')

    return xvec, CONVG



# this is copied from scipy.linalg, to make compatible with jax.numpy
def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.
    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row.  If r is not given, ``r == conjugate(c)`` is
    assumed.
    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.
    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
    See also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0.  The behavior in previous
    versions was undocumented and is no longer supported.
    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])
    """
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1D array of values to be used in the matrix, containing a reversed
    # copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
    indx = a + b
    # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
    # that `vals[indx]` is the Toeplitz matrix.
    return vals[indx]
