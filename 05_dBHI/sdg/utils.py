import numpy as np
import scipy
from numba import njit  # For compiling at run-time
from scipy.signal.windows import kaiser


# Utility functions not specific to the class
def convolve_same(A, B):
    """
    This function replicates the behaviour of Matlab convolve function, with the "same" option enabled
    Python also has a np.convolve function with the optional "same" parameter. However this
    shifts the input relative to matlab output if the second input parameter is even.
    for example
    a = [5, 4, 3, 2, 1]
    b = [2, 2, 2, 2]
    full (a,b)-convolution = [2, 6, 12, 20, 28, 24, 24, 18, 10]
    np.convolve(a, b, "same") => [6, 12, 20, 28, 24]
    (matlab) conv(a, b, "same") => [12, 20, 28, 24, 18]

    To make the function comparable to matlab, the special case is considered
    """
    if len(B) % 2 == 0:
        conv_result = np.convolve(A, B)
        start_idx = int(np.ceil((len(conv_result) - len(A)) / 2))
        return conv_result[start_idx : start_idx + len(A)]
    else:
        return np.convolve(A, B, "same")


def arx_least_squares(y, u, na=1, nb=1, nk=1):
    """
    Fit ARX model: y[t] + a1*y[t-1] = b1*u[t-nk]
    Returns: a1, b1
    """
    N = len(y)
    # Build regression matrix
    Y = y[max(na, nb + nk - 1) :]
    Phi = []
    for t in range(max(na, nb + nk - 1), N):
        row = []
        # Output lags
        for i in range(1, na + 1):
            row.append(-y[t - i])
        # Input lags (with delay nk)
        for j in range(nb):
            row.append(u[t - nk - j])
        Phi.append(row)
    Phi = np.array(Phi)
    # Least squares solution
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    # Computing error term
    Y_pred = Phi @ theta
    residuals = Y - Y_pred

    dof = len(residuals) - (na + nb)
    variance = np.sum(np.power(residuals, 2)) / dof
    return (*theta, variance)


@njit
def _arx_solve_compiled(y, u, na, nb, nk):
    N = y.shape[0]
    start = na if na > (nb + nk - 1) else (nb + nk - 1)
    rows = N - start
    cols = na + nb
    if rows <= 0:
        return np.empty(0, dtype=np.float64), 0.0, start

    Phi = np.empty((rows, cols), dtype=np.float64)
    Y = np.empty(rows, dtype=np.float64)

    for idx in range(rows):
        t = idx + start
        Y[idx] = y[t]
        # AR lags: -y[t-1], -y[t-2], ...
        for i in range(na):
            Phi[idx, i] = -y[t - (i + 1)]
        # Input lags with delay nk
        for j in range(nb):
            Phi[idx, na + j] = u[t - (nk + j)]

    # Normal equations: (Phi^T Phi) theta = Phi^T Y
    A = Phi.T.dot(Phi)
    b = Phi.T.dot(Y)
    # Solve linear system (assumes A is invertible / well-conditioned)
    theta = np.linalg.solve(A, b)

    # residuals and variance
    Y_pred = Phi.dot(theta)
    s = 0.0
    for i in range(rows):
        diff = Y[i] - Y_pred[i]
        s += diff * diff
    dof = rows - (na + nb)
    if dof > 0:
        variance = s / dof
    else:
        variance = 0.0
    return theta, variance, start


def arx_least_squares_fast(y, u, na=1, nb=1, nk=1):
    """
    Compiled ARX via numba (_arx_solve_compiled).
    Returns: (a1, b1, variance) for na=1,nb=1 (or theta elements and variance)
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    try:
        theta, variance, _ = _arx_solve_compiled(y, u, na, nb, nk)
    except:
        return [0, 0, 0]
    if theta.size == 0:
        # not enough data; return zeros and zero variance
        return tuple([0.0] * (na + nb)) + (0.0,)

    # Return parameters plus variance to match previous API
    # previous code returned (*theta, variance)
    out = tuple(theta.tolist()) + (variance,)
    return out


def SPWVD(signal, fs):
    # Make even
    if len(signal) % 2 == 1:
        signal = np.append(signal, 0)
    # In matlab if the input is complex the data, an zeros are appended equivalently to the length
    signal = np.append(signal, np.zeros(len(signal)))

    # Change to offer an input value
    DataLen = len(signal)
    FreqBins = int(DataLen / 2)

    # Smoothing windows
    nTimeWin = round(DataLen / 10)
    if nTimeWin % 2 == 0:
        nTimeWin += 1
    timeWindow = kaiser(nTimeWin, 20)

    nFreqWin = np.min([FreqBins, np.round(DataLen / 4)])
    if nFreqWin % 2 == 0:
        nFreqWin += 1
    freqWindow = kaiser(nFreqWin, 20)

    # time vector
    totalDuration = 0.5 * DataLen / fs
    timeVec = np.linspace(0, totalDuration, DataLen)

    ## In the wvdImpl file
    freqVec = np.linspace(0, 1, FreqBins)
    N2 = DataLen
    N = int(np.ceil(DataLen / 2))
    # Don't know yet what these will be used for
    nhTime = int(np.floor(DataLen / 2))
    nhFreq = int(np.ceil(FreqBins / 2))

    ## In the getSpwvdWindowParams file
    nhFreqWin = int(np.floor(nFreqWin / 2))
    maxLag = int(np.floor(nTimeWin / 2))

    G1 = np.roll(freqWindow, np.ceil(len(freqWindow) / 2))
    g2 = np.roll(timeWindow, np.ceil(len(timeWindow) / 2))
    ##

    nVec = np.linspace(0, N - 1, N)
    mVec = np.linspace(0, maxLag, maxLag + 1)
    M = maxLag + 1

    nVecMN = np.matlib.repmat(nVec.reshape(1, -1), M, 1)
    mVecMN = np.matlib.repmat(np.reshape(np.floor(mVec / 2), (-1, 1)), 1, N)
    posLagMat = np.mod(
        nVecMN + np.matlib.repmat(np.ceil(np.reshape(mVec, (-1, 1)) / 2), 1, N), N2
    ).astype(int)
    negLagMat = np.mod(nVecMN - mVecMN, N2).astype(int)

    autocorr = np.multiply(signal[posLagMat], np.conjugate(signal[negLagMat]))
    # --- applying time smoothing
    mnMat = np.multiply(np.matlib.repmat(np.reshape(g2[0 : maxLag + 1], (-1, 1)), 1, N), autocorr)

    #  --- Applying frequency smoothing
    afMat = scipy.fft.fft(mnMat, axis=1)

    # Indices
    noDoppStart = int(nhFreqWin + 1)
    noDoppStop = int(nhTime - nhFreqWin - 1 - (np.remainder(nFreqWin, 2) - 1))
    noDoppSupportIxs = np.linspace(noDoppStart, noDoppStop, noDoppStop - noDoppStart + 1).astype(
        int
    )
    posDoppIxs = np.linspace(0, nhFreqWin, nhFreqWin + 1).astype(int)
    negDoppIx = nhFreqWin - (nFreqWin % 2 == 0).astype(int)

    # Matrix to contain the filtered frequency spectrum
    afWinMat = np.zeros((maxLag + 1, nhTime), dtype=complex)
    afMatRow = afMat.shape[0]

    # filtering
    afWinMat[:, posDoppIxs] = np.multiply(
        np.matlib.repmat(np.reshape(G1[posDoppIxs], (1, -1)), afMatRow, 1), afMat[:, posDoppIxs]
    )
    afWinMat[:, -1 - negDoppIx :] = np.multiply(
        np.matlib.repmat(np.reshape(G1[-1 - negDoppIx :], (1, -1)), afMatRow, 1),
        afMat[:, -1 - negDoppIx :],
    )
    afWinMat[:, noDoppSupportIxs] = 0

    # Return to time-lag domain
    mnMat = scipy.fft.ifft(afWinMat, axis=1)

    # Due to symmetry in autocorrelation we can retain only half without loss of information
    kmat = np.zeros((2 * FreqBins, nhTime), dtype=complex)
    kmat[0 : maxLag + 1, :] = mnMat[0 : maxLag + 1, 0:nhTime]

    # Use time-lag function symmetry to obtain (-) lag vals from (+) vals.
    mEvenIxs = np.linspace(1, nhFreq - 1, nhFreq - 1).astype(int)
    mOddIxs = np.linspace(0, nhFreq - 1, nhFreq).astype(int)
    # Use time-lag function symmetry to obtain (-) lag vals from (+) vals.
    kmat[[2 * FreqBins - 2 * mEvenIxs], :] = np.conjugate(kmat[2 * mEvenIxs, :])
    kmat[[2 * FreqBins - 2 * mOddIxs - 1], :] = np.conjugate(kmat[2 * mOddIxs + 1, :])

    # Convert to Time-frequency domain with FT m->k
    wvdMat = np.zeros((FreqBins, N2), dtype=complex)

    # "Even lag values are conjugate symmetric. Here we calculate WVD[k,2n] = FT{K[2m,n]}"
    wvdMat[:, 0::2] = scipy.fft.fft(kmat[0::2, :], axis=0)

    kMatPrimeOdd = kmat[1::2, :]
    nFreqOdd = int(kMatPrimeOdd.shape[0])
    nhFreqOdd = int(np.ceil(nFreqOdd / 2))
    kMatHat = np.zeros((nFreqOdd, nhTime), dtype=complex)
    kMatHat[0, :] = np.imag(kMatPrimeOdd[0, :])
    # frexIxs1 = np.linspace()
    # Calculate (1/2j) * [ K[2m+1,n] - K*[2*Nf-2m-1,n] ] for nonzero m.
    kMatHat[1 : (nhFreqOdd + 1), :] = (
        1
        / (2 * 1j)
        * (
            kMatPrimeOdd[1 : (nhFreqOdd + 1), :]
            - np.conjugate(kMatPrimeOdd[nFreqOdd::-1, :][: nFreqOdd - nhFreqOdd, :])
        )
    )
    kMatHat[(nhFreqOdd + 1) :, :] = np.conjugate(kMatHat[1:nhFreqOdd, :][::-1, :])

    tfMat = scipy.fft.fft(kMatHat, axis=0)
    k = np.linspace(1, FreqBins - 1, FreqBins - 1).astype(int)
    A = np.cos(np.pi / FreqBins * k)
    B = np.sin(np.pi / FreqBins * k)

    wvdMat[k, 1::2] = (
        np.matlib.repmat(np.reshape((A * A + B * B) / B, (-1, 1)), 1, tfMat.shape[1]) * tfMat[k, :]
    )
    wvdMat[0, 1::2] = np.sum(kmat[1::2, :], axis=0)
    #
    freqVec = freqVec * fs / 2
    return freqVec, timeVec, np.real(wvdMat)


def integrate_frequency_band(TFR, f1, f2, f):
    # Integrates over frequency band spanned by f1 and f2
    idx1 = np.where(f1 <= f)[0][0]
    idx2 = np.where(f <= f2)[0][-1] + 1
    return scipy.integrate.trapezoid(TFR[idx1:idx2, :], axis=0)


def interp1cubic(x, y, xq, a=-0.5):
    """
    1D cubic convolutional interpolation function matching MATLAB's interp1 with 'cubic' option.

    Uses Catmull-Rom cubic convolution in the interior and quadratic (Lagrange) interpolation
    at the edges where a full 4-point stencil is not available. This matches MATLAB's behavior.

    Parameters:
    -----------
    x : array_like
        Known x values (1D array)
    y : array_like
        Known y values (1D array)
    xq : array_like
        Query x values (1D array)
    a : float, optional
        Kernel parameter for Keys' cubic kernel (default: -0.5 for Catmull-Rom)

    Returns:
    --------
    yq : ndarray
        Interpolated y values at xq.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64).ravel()

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("Need at least 3 points for interpolation")

    # Cubic convolution kernel (Keys' method)
    def cubic_kernel(s):
        """Cubic convolution kernel weight function"""
        s = np.abs(s)
        w = np.zeros_like(s)

        # For |s| <= 1
        mask1 = s <= 1
        w[mask1] = (a + 2) * s[mask1] ** 3 - (a + 3) * s[mask1] ** 2 + 1

        # For 1 < |s| <= 2
        mask2 = (s > 1) & (s <= 2)
        w[mask2] = a * s[mask2] ** 3 - 5 * a * s[mask2] ** 2 + 8 * a * s[mask2] - 4 * a

        return w

    def quadratic_interp(x_pts, y_pts, xq_val):
        """Lagrange quadratic interpolation using 3 points"""
        x0, x1, x2 = x_pts
        y0, y1, y2 = y_pts

        # Lagrange basis polynomials
        L0 = ((xq_val - x1) * (xq_val - x2)) / ((x0 - x1) * (x0 - x2))
        L1 = ((xq_val - x0) * (xq_val - x2)) / ((x1 - x0) * (x1 - x2))
        L2 = ((xq_val - x0) * (xq_val - x1)) / ((x2 - x0) * (x2 - x1))

        return L0 * y0 + L1 * y1 + L2 * y2

    yq = np.zeros(len(xq))

    for i, xqi in enumerate(xq):
        # Handle extrapolation (return NaN for out of bounds)
        if xqi < x[0] or xqi > x[-1]:
            yq[i] = np.nan
            continue

        # Find the interval containing xqi
        idx = np.searchsorted(x, xqi, side="right") - 1

        # Use quadratic interpolation at the edges (first and last intervals)
        # This matches MATLAB's behavior when there aren't enough points for cubic
        if idx == 0:
            # First interval: use first 3 points
            yq[i] = quadratic_interp(x[0:3], y[0:3], xqi)
        elif idx == len(x) - 1:
            # Last interval: use last 3 points
            yq[i] = quadratic_interp(x[-3:], y[-3:], xqi)
        else:
            # Interior: use cubic convolution with 4 points
            idx = np.clip(idx, 1, len(x) - 3)  # Ensure we have 2 points on each side

            # Get the 4 neighboring points (p-1, p, p+1, p+2)
            # where p is the point just before xqi
            i0 = idx - 1
            i1 = idx
            i2 = idx + 1
            i3 = idx + 2

            # Get x-coordinates of the 4 neighbors
            x0, x1, x2, x3 = x[i0], x[i1], x[i2], x[i3]
            y0, y1, y2, y3 = y[i0], y[i1], y[i2], y[i3]

            # Compute the average spacing for normalization
            # This is important for non-uniformly spaced data
            dx_avg = (x3 - x0) / 3.0
            if dx_avg == 0:
                yq[i] = y[i1]
                continue

            # Compute normalized distances from query point to each neighbor
            # These should be in units of the local spacing
            s0 = (xqi - x0) / dx_avg
            s1 = (xqi - x1) / dx_avg
            s2 = (xqi - x2) / dx_avg
            s3 = (xqi - x3) / dx_avg

            s = np.array([s0, s1, s2, s3])

            # Get weights from cubic kernel
            weights = cubic_kernel(s)

            # Normalize weights to ensure they sum to 1
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum

            # Interpolate
            yq[i] = weights[0] * y0 + weights[1] * y1 + weights[2] * y2 + weights[3] * y3

    return yq


# ^^^^^^^^^^^^^^^^^^^^^^^^^^ Non-class specific support functions ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
