import inspect  # To analyze the optional input arx and wvd functions
from warnings import warn

import numpy as np
import scipy
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import (
    SPWVD,
    arx_least_squares,
    arx_least_squares_fast,
    convolve_same,
    integrate_frequency_band,
    interp1cubic,
)


class PySDG:
    def __init__(
        self,
        eeg: np.array = None,
        eeg_fs: float = None,
        r_peaks: np.array = None,
        RR: np.array = None,
        t_RR: np.array = None,
        RR_int: np.array = None,
        t_RR_int: np.array = None,
        fs_RR_int: float = None,
        channels: list = None,
        tfr_t: np.array = None,
        tfr_eeg: np.array = None,
        tfr_hrv: np.array = None,
        flims_eeg: dict = {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [12, 32]},
        fs_bhi: float = 4,
        window: float = 4,
        parallelize: bool = True,
        progressbar: bool = True,
        wvd_func=None,
        arx_func=None,
        arx_orders=[1, 1, 1],
        cs: float = 0,
        cp: float = 0,
    ):
        """
        eeg: numpy.array        - of shape [Nc, Nt], where Nc corresponds channels and Nt corresponds to timesamples
        eeg_fs: float           - The sample frequency at which the eeg has been acquired
        r_peaks: numpy.array    - Timepoints for detected R-peaks, provided in seconds.
                                  The timepoints are relative to the first sample of eeg
                                  Alternatively provide the r_peaks-derived values below.
            vvvvvvvvvvvvvvvvvvvvvvvvvvvv If r_peaks is not provided vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            RR: numpy.array         - series of RR-intervals recorded for each R-peak
            t_RR: numpy.array       - Timeseries corresponding to each value of RR
            RR_int: numpy.array     - Uniformly, interpolated RR-values, sampled at a frequency of fs_RR_int
            t_RR_int: numpy.array   - Timeseries corresponding to each value of RR_int
            fs_RR_int: float        - Frequency for sampling the interpolated RR-values
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ If r_peaks is not provided ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        tfr_t, tfr_eeg and tfr_hrv, offers the user the option of manually providing time-frequency response for the SDG computation
        This would be an alternative to using the member methods compute_tfr_eeg and compute_tfr_hrv
        vvvvvvvvvvvvvvvvvvvvvvvvvvvv optional vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        tfr_t: numpy.array      - (optional) shape [Nt,] Timeseries corresponding to the tfr_eeg and tfr_hrv
        tfr_eeg: numpy.array    - (optional) shape [Nc, Nt, Neb] time-frequency response of the EEG signal
            First index is channel, second is the time index, third is frequency band index. The third index indicates the integration
            over the a frequency band of the PSD, at that particular timepoint for that particular channel
        tfr_hrv: numpy.array    - (optional) shape [Nhb, Nt], time-frequency response of the HRV signal
            First index corresponds to the frequency band index. Default would be integrated over LF and HF frequency bands
            Second index corresponds to the timepoint.
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ optional ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        flims_eeg: dictionary   - Keys corresponds to band names, and values should be a list with lower- and upper frequency boundary in Hz to be analyzed
        fsi_bhi: float          - BHI sampling frequency, typically chosen to be <10Hz
        parallelize: bool       - Flag whether internal SDG computation should be run in parallel on available cores

        Custom functions can optionally be passed as parameters to operate on internal computations.
        vvvvvvvvvvvvvvvvvvvvvvvvvvvv optional vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        wvd_func: function      - function to be called on the analytical interpolated RR signal. This function must take two inputs and provide three outputs as follows
            f, t, wvd = wvd_func(signal, fs) - wvd: np.array [Nf, Nt], is a powerspectral density estimated by the function
                                                f: np.array [Nf,] is  the a series of frequencies in increasing order, corresponding to the rows of wvd
                                                t: np.array [Nt,] is  the a timeseries corresponding to the columns of wvd
                                                fs: float sampling frequency of the input signal
            wvd_func operates inside the compute_tfr_hrv()
        arx_func: function      - function called estimating the ARX parameters. This function should output three parameters and take the parameters as follows
            a1, x1, sigma2 = arx_least_squares_fast(y, u, na, nb, nk)   - a1, x1 are the respective autoregressive and exogenous parameters approximated in the arx model,
                                                                        - sigma2 is the residual noise variance when subtracting the model the signal
                                                                        - y is the tfr signal of a particular channel
                                                                        - u is the tfr signal of a particular hrv band (i.e. LF or HF)
                                                                        - na, nb and nk are polynomial orders of the model being estimated, provided through the arx_orders parameter
        window: float   - The instantaneous window size in seconds for computing the SDG parameters. Default value is 4.
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ optional ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cs, cp: float       - constant sympathetic and parasympathetic terms
        """
        # EEG parameters
        self.eeg = dict()
        self.eeg["data"] = eeg
        self.eeg["fs"] = eeg_fs
        self.eeg["channels"] = channels

        # HRV parameters
        self.hrv = dict()
        # User should either provide all the R_peak derived series or the R_peak series itself
        if (
            (RR is None)
            | (t_RR is None)
            | (t_RR_int is None)
            | (RR_int is None)
            | (fs_RR_int is None)
        ):
            if r_peaks is None:
                warn(
                    "PySDG.__init__: to run the SDG pipeline, the class needs to be initialized with a vector of the r_peak location or all derived parameters"
                )
            else:
                fs_RR_int = fs_bhi
                RR = np.diff(r_peaks)
                t_RR = r_peaks[1:]
                t_RR_int = np.arange(t_RR[0], t_RR[-1], 1 / fs_RR_int)
                interpolator = scipy.interpolate.CubicSpline(t_RR, RR)
                RR_int = interpolator.__call__(t_RR_int, extrapolate=False)
        self.hrv["t"] = t_RR  ### Not used
        self.hrv["data"] = RR  # This is needed for CPr and CSr computation as well
        self.hrv["t_int"] = t_RR_int
        self.hrv["data_int"] = RR_int
        self.hrv["fs_hrv_int"] = fs_RR_int

        # TFR parameters - These inputs can alternatively be provided if not computed using the class methods with the EEG- and HRV parameters
        self.tfr = dict()
        self.tfr["t"] = tfr_t
        self.tfr["eeg"] = tfr_eeg
        self.tfr["hrv"] = tfr_hrv

        # BHI specific parameters and SDG output
        self.bhi = dict()  # Placeholder for SDG output
        self.bhi["parameters"] = dict()
        self.bhi["fs_bhi"] = fs_bhi  # Samples per second of BHI to be computed
        self.bhi["flims_eeg"] = (
            flims_eeg  # a dict() of names holding lists with upper- and lower level of frequency bands to be analyzed
        )

        ## Computational options
        self.opts = dict()
        ## The TFR function to be used for HRV
        # Check that the provided function takes a numpy array and a float as input and provides three outputs
        if wvd_func is not None:
            sig = np.array([0.0, 1.0, 0.0])
            fs_test = 4.0
            sig_outs = inspect.signature(wvd_func).parameters
            if len(sig_outs) != 2:
                raise ValueError(
                    "PySDG.__init__: provided wvd_func must take two input parameters: signal (numpy.array) and fs (float)"
                )
            test_out = wvd_func(sig, fs_test)
            if not (isinstance(test_out, tuple) and len(test_out) == 3):
                raise ValueError(
                    "PySDG.__init__: provided wvd_func must provide three outputs: f (numpy.array), t (numpy.array), wvd (numpy.array)"
                )
        else:
            wvd_func = SPWVD
        self.opts["WVD function"] = wvd_func
        # Related to SDG related computation
        self.opts["parallelize"] = (
            parallelize  # Option to enable parallelization of the inner bhi method
        )
        self.opts["progressbar"] = progressbar  # Option to enable progressbar for SDG computation
        ## The ARX function to be used for SDG
        # Check that the provided function takes two numpy arrays and three integers as input and provides three outputs
        if arx_func is not None:
            y_test = np.array([0.0, 1.0, 0.0])
            u_test = np.array([0.0, 1.0, 0.0])
            na_test, nb_test, nk_test = 1, 1, 1
            arx_outs = inspect.signature(arx_func).parameters
            if len(arx_outs) != 5:
                raise ValueError(
                    "PySDG.__init__: provided arx_func must take five input parameters: y (numpy.array), u (numpy.array), na (int), nb (int), nk (int)"
                )
            test_out = arx_func(y_test, u_test, na_test, nb_test, nk_test)
            if not (isinstance(test_out, tuple) and len(test_out) == 3):
                raise ValueError(
                    "PySDG.__init__: provided arx_func must provide three outputs: a1 (float), b1 (float), sigma2 (float)"
                )
        else:
            arx_func = arx_least_squares_fast
        self.opts["arx function"] = arx_func
        self.opts["arx orders"] = arx_orders

        # Window size for evaluating BHI model - window provided in second
        if window * fs_bhi < 15:  # Must be at least 15 samples
            window = np.ceil(15 / fs_bhi)
        self.opts["wind"] = int(
            window * fs_bhi
        )  # self.opts["wind"] is provided in samples, window is provided in seconds
        self.opts["Cs"] = cs
        self.opts["Cp"] = cp

    def compute_tfr_eeg(self, window=2):
        """
        Input:
            - window: Window used at each time step for the short-time fourier transform
        Output:
            - self.tfr["eeg"]:      [Nch x Nl x Nb]-dimensional numpy.array, where
                                        Nch corresponds to the EEG channels
                                        Nl corresponds to the timepoints at which the stft is computed
                                        Nb corresponds to the frequency bands in self.bhi["flims_eeg"]
            - self.tfr["t"]:        timepoints corresponding to second dimension of self.tfr["eeg"]
                                    timepoints are centered at the window used to compute TFR
        """

        Nbands = len(self.bhi["flims_eeg"])
        Nch = int(self.eeg["data"].shape[0])

        w = int(self.eeg["fs"] * window)  # window length in #samples
        step = round(
            self.eeg["fs"] / self.bhi["fs_bhi"]
        )  # Step size in #samples between consecutive windows

        # TFR EEG
        n = self.eeg["data"].shape[1]  # Length of signal
        l = int(np.floor((n - w) / step))  # number of segments
        TFR_EEG = np.zeros([Nch, l, Nbands])  # Pre-allocation
        i_b_val = 0
        for i_b in self.bhi["flims_eeg"]:  # key values of dictionary
            for i_l in range(l):
                TFR_EEG[:, i_l, i_b_val] = self._bandpower(
                    self.eeg["data"][:, (i_l * step) : (i_l * step + w)],
                    fs=self.eeg["fs"],
                    fmin=self.bhi["flims_eeg"][i_b][0],
                    fmax=self.bhi["flims_eeg"][i_b][1],
                )
            i_b_val += 1

        t_tfr_eeg = np.arange(0, l) * step / self.eeg["fs"] + window / 2
        self.tfr["t"] = t_tfr_eeg
        self.tfr["eeg"] = TFR_EEG
        self.bhi["parameters"]["EEG_TFR_step"] = step

    def compute_tfr_hrv(self):
        """
        Output:
            - self.tfr["hrv"]:      [Nhb x Nl]-dimensional numpy.array, where
                                        Nhb corresponds to the hrv band (i.e. LF and HF)
                                        Nl correpsonds to the time points of self.tfr["t"], the hrv band series is interpolated to these timepoints
            - self.tfr["eeg"]:      updated to only be represented at points where self.tfr["hrv"] is not nan
            - self.tfr["t"]:        updated to only be represented at points where self.tfr["hrv"] is not nan
        """

        f, t_tfr, wvd = self.opts["WVD function"](
            scipy.signal.hilbert(self.hrv["data_int"] - np.mean(self.hrv["data_int"])),
            fs=self.bhi["fs_bhi"],
        )
        t_tfr_rr = np.linspace(
            self.hrv["t"][0], self.hrv["t"][-1], wvd.shape[1]
        )  # Custom time scale is made to match MATLAB implementation
        # HF upper bound raised from the adult 0.40 to 0.80 Hz for children, to
        # match preprocessing/ecg/compute_hrv_timeseries.py (HF_BAND) so both HRV pipelines in
        # this project use one definition. Children breathe faster: in this
        # cohort the RSA peak sits at a median 0.317 Hz, 6% of subjects peak
        # above 0.40 Hz, and the power the adult ceiling discards is confounded
        # with age (24.7% at 5 yr vs 14.6% at 10 yr, rho=-0.227, p=0.0055).
        # Requires fs_bhi >= 2 for headroom (fs_bhi sets the RR interpolation
        # rate, so the spectrum only reaches fs_bhi/2); dbhi_params uses 4.
        LF = integrate_frequency_band(wvd, 0.04, 0.15, f)
        HF = integrate_frequency_band(wvd, 0.15, 0.80, f)

        TFR_HRV = np.zeros([2, len(self.tfr["t"])])

        TFR_HRV[0, :] = interp1cubic(t_tfr_rr, LF, self.tfr["t"])
        TFR_HRV[1, :] = interp1cubic(t_tfr_rr, HF, self.tfr["t"])

        # The RR series is cubic-spline interpolated onto a uniform grid before
        # the SPWVD, so a stretch with no detected beats still comes out with
        # finite LF/HF here -- values fabricated by interpolating across the
        # hole, not measured. Mark any gap longer than the SDG window as
        # missing. This is the same threshold that already leaves CPr/CSr NaN
        # in _compute_cpr_and_csr, so bth and htb end up marked by one rule
        # instead of two. RR[i] is the interval ENDING at t_RR[i], so the
        # unsupported span is (t_RR[i]-RR[i], t_RR[i]).
        _t_rr = np.asarray(self.hrv["t"]).ravel()
        _rr = np.asarray(self.hrv["data"]).ravel()
        _gap_s = self.opts["wind"] / self.bhi["fs_bhi"]
        for _g in np.where(_rr > _gap_s)[0]:
            TFR_HRV[:, (self.tfr["t"] > _t_rr[_g] - _rr[_g]) & (self.tfr["t"] < _t_rr[_g])] = np.nan

        # Strip NaN at the EDGES only, where the HRV TFR has no support at all.
        # An INTERIOR run of NaN means the R-peak series has a gap: deleting
        # those columns would splice the two sides together, and the ARX window
        # and convolve_same downstream would then smear across a discontinuity
        # as if it were continuous time. Keep them so the gap stays marked.
        _valid = ~np.any(np.isnan(TFR_HRV), axis=0)
        Ind_toKeep = np.zeros_like(_valid)
        if _valid.any():
            Ind_toKeep[_valid.argmax() : len(_valid) - _valid[::-1].argmax()] = True
        # Ind_toKeep = np.where(np.logical_and(np.logical_not(np.isnan(TFR_HRV[0,:])), np.logical_not(np.isnan(TFR_HRV[1,:]))))[0]
        self.tfr["t"] = self.tfr["t"][Ind_toKeep]
        self.tfr["eeg"] = self.tfr["eeg"][:, Ind_toKeep, :]
        self.tfr["hrv"] = TFR_HRV[:, Ind_toKeep]

    def compute_sdg(self, interpmethod=None):
        """
        Input:
            - interpmethod, string (optional) specifies what internal interpolation method should be used in the internal CPr CSr estimation, options are:
                "PChip" (default), "Akima", "CubicSpline", "CubicConv"
        Output:
            - self.bhi["bth"]: numpy.array  [Nb, Nhb, Nc, Nt1] - The computed instantaneous brain-to-heart coupling
                Nb is the EEG frequency bands, NHb is the hrv frequency bands (i.e. LF and Hf)
                Nc is the eeg channel index, Nt1 is the time index corresponding to self.bhi["t"]["time_BToH"]
            - self.bhi["htb"]: numpy.array  [Nb, Nhb, Nc, Nt2] - The computed instantaneous heart-to-brain coupling
                Nb is the EEG frequency bands, NHb is the hrv frequency bands (i.e. LF and Hf)
                Nc is the eeg channel index, Nt2 is the time index corresponding to self.bhi["t"]["time_HToB"]
            - self.bhi["t"]  : dict with numpy.arrays
                self.bhi["t"]["time_BToH"] is time-series corresponding to the timepoints of self.bhi["bth"]
                self.bhi["t"]["time_HToB"] is time-series corresponding to the timepoints of self.bhi["htb"]
        """
        Nbands = len(self.bhi["flims_eeg"])
        win_RR = 15  # number RR samples
        first_iter = True
        Nrr_bands = 2
        # Making a linear combinations of EEG and HRV bands for the progressbar
        linear_index = [(eeg_b, rr_b) for eeg_b in range(Nbands) for rr_b in range(Nrr_bands)]
        for i in tqdm(
            range(len(linear_index)),
            desc="Computing SDG Brain-Heart Interplay",
            disable=not self.opts["progressbar"],
        ):
            eeg_b, rr_b = linear_index[i]
            tmpHtB, tmpBtH0, tmpBtH1, time_bhi, window = self._bhi_model_wT(
                eeg_b, rr_b, win_RR, TV=True, interpmethod=interpmethod
            )[0:5]
            if first_iter:  # Pre-allocation
                HtB = np.zeros([Nbands, Nrr_bands, *tmpHtB.shape])
                BtH = np.zeros([Nbands, 2, *tmpBtH0.shape])
                first_iter = False
            HtB[eeg_b, rr_b, :, :] = tmpHtB
            BtH[eeg_b, 0, :, :] = tmpBtH0
            BtH[eeg_b, 1, :, :] = tmpBtH1

        self.bhi["bth"] = BtH
        self.bhi["htb"] = HtB
        self.bhi["t"] = time_bhi

    def _bandpower(
        self, data: np.array, fs=None, fmin=None, fmax=None
    ):  # Compute the bandpower of the signal data in a specific frequency band
        # Source inspiration: https://stackoverflow.com/questions/44547669/python-numpy-equivalent-of-bandpower-from-matlab, https://stackoverflow.com/a/44915647
        f, Pxx = scipy.signal.periodogram(data, fs=fs, window="hamming", detrend=False)

        # If only one channel we have to ensure that the time index is aligned with the second dimension
        if data.shape[0] == 1:
            Pxx.reshape([1, -1])
            f.reshape([1, -1])
        ind_min = np.squeeze(np.where(f <= fmin))[-1]
        ind_max = np.squeeze(np.where(f >= fmax))[0] + 1  # open range indexing

        # This is implemented according to the matlab procedure for equivalence comparison
        # Small differences occur between the periodogram methods
        df = np.diff(f)
        missingWidth = (f[-1] - f[0]) / (len(f) - 1)
        df = np.append(df, missingWidth)
        return np.dot(Pxx[:, ind_min:ind_max], df[ind_min:ind_max])

    def _bhi_model_wT(self, eeg_b, rr_b, win_RR, TV, interpmethod=None):
        """
        Input:
            - self:                 PySDG class instance  which has run compute_tfr_eeg() and compute_tfr_hrv()
            - eeg_b:                index of the eeg frequency band to be analyzed
            - rr_b:                 index of the hrv frequency band to be analyzed
            - win_RR:               window length in number of RR samples for the computation of CPr and CSr
            - TV:                   boolean flag whether time-varying BHI should be computed
            - interpmethod:         string (optional) specifies what internal interpolation method should be used in the internal CPr CSr estimation, options are:
                                        "PChip" (default), "Akima", "CubicSpline", "CubicConv", "CubicConvMatlab", "CubicConvKeys"
        Output:
            - HeartToBrain:                 numpy.array [Nc, Nt1] - The computed instantaneous heart-to-brain coupling for the specified eeg and hrv band
            - BrainToLF:                    numpy.array [Nc, Nt2] - The computed instantaneous brain-to-heart coupling for the specified eeg and hrv band (LF)
            - BrainToHF:                    numpy.array [Nc, Nt2] - The computed instantaneous brain-to-heart coupling for the specified eeg and hrv band (HF)
            - time_bhi:                     dict with numpy.arrays - Time indices corresponding to the coupling indices
                - time_bhi["time_HToB"]:    numpy.array - timepoints corresponding to HeartToBrain
                - time_bhi["time_BToH"]:    numpy.array - timepoints corresponding to BrainToLF and BrainToHF
            - HeartToBrain_sigma:           numpy.array - Model parameter for fitting evaluation
            - HeartToBrain_mc:              numpy.array - Model parameter for fitting evaluation

        Description:
            This software assumes that input series
            are all artifact free, e.g., heartbeat dynamics free of algotirhmic and/or physiological artifacts; e.g.
            EEG series free of artifacts from eye blink, movement, etc.
        ---------------------------------------------------------------------------------------------
         This code implements the theoretical dissertation published in:
         [1] Catrambone Vincenzo, Alberto Greco, Nicola Vanello, Enzo Pasquale Scilingo,
         and Gaetano Valenza. "Time-Resolved Directional Brain–Heart Interplay Measurement
         Through Synthetic Data Generation Models."
         Annals of biomedical engineering 47, no. 6 (2019): 1479-1489.

        With the development and code base initially published in:
         [2] ... TBD
        ---------------------------------------------------------------------------------------------
        Copyright (C) 2019 Vincenzo Catrambone, Gaetano Valenza, TBD

        This program is a free software; you can redistribute it and/or modify it under
        the terms of the GNU General Public License as published by the Free Software
        Foundation; either version 3 of the License, or (at your option) any later
        version.

        If you use this program in support of published research, please include a
        citation of the references above. If you use this code in a software package,
        please explicitly inform the end users of this copyright notice and ask them
        to cite the references above in their published research.
        ---------------------------------------------------------------------------------------------
        """
        # %% Testing of the input

        [Nch, Nt] = [self.tfr["eeg"].shape[0], self.tfr["eeg"].shape[1]]

        TFR_EEG = self.tfr["eeg"][:, :, eeg_b].reshape([Nch, Nt])
        TFR_HRV = np.squeeze(self.tfr["hrv"][rr_b, :])
        time_tfr = self.tfr["t"]

        CPr, CSr = self._compute_cpr_and_csr(win_RR, interpmethod=interpmethod)

        # nan-aware min/max: a marked gap must not turn the whole normalised
        # series NaN. Matches the CPr/CSr normalisation, which is already
        # nanmin/nanmax. No-op when there is nothing marked.
        TFR_EEG = np.divide(
            TFR_EEG - np.expand_dims(np.nanmin(TFR_EEG, axis=-1), 1),
            np.expand_dims((np.nanmax(TFR_EEG, axis=-1) - np.nanmin(TFR_EEG, axis=-1)), 1),
        )
        TFR_HRV = np.divide(
            TFR_HRV - np.nanmin(TFR_HRV, axis=-1),
            (np.nanmax(TFR_HRV, axis=-1) - np.nanmin(TFR_HRV, axis=-1)),
        )

        # Pre-allocation
        Niter = int(
            np.min(
                [len(CPr), TFR_EEG.shape[1] - self.opts["wind"], len(TFR_HRV) - self.opts["wind"]]
            )
        )
        nonan = np.where(np.logical_not(np.isnan(CPr)))[0].shape[
            0
        ]  # For some parameters the inside model removes the nan positions equivalent to CPr and CSr
        HeartToBrain = np.zeros([Nch, Niter])
        BrainToLF = np.zeros([Nch, nonan])
        BrainToHF = np.zeros([Nch, nonan])
        HeartToBrain_sigma = np.zeros([Nch, Niter])
        HeartToBrain_mc = np.zeros([Nch, Niter])
        # time_bhi = [0]*Nch

        if Nch > 1 and self.opts["parallelize"]:
            out = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(self._BHI_InsideModel)(TFR_EEG[ch, :], TFR_HRV, CPr, CSr, time_tfr, TV)
                for ch in range(Nch)
            )
            HTB_vals, BLF_vals, BHF_vals, H_sig_vals, H_mc_vals, t_bhi_vals = zip(*out)
            HeartToBrain = np.stack(HTB_vals, axis=0)
            BrainToLF = np.stack(BLF_vals, axis=0)
            BrainToHF = np.stack(BHF_vals, axis=0)
            HeartToBrain_sigma = np.stack(H_sig_vals, axis=0)
            HeartToBrain_mc = np.stack(H_mc_vals, axis=0)
            time_bhi = t_bhi_vals[0]
        else:
            # serial path: collect + stack (mirror the parallel branch) so the
            # actual returned lengths are used, avoiding the Niter preallocation
            # mismatch that breaks for window != default.
            out = [
                self._BHI_InsideModel(TFR_EEG[ch, :], TFR_HRV, CPr, CSr, time_tfr, TV)
                for ch in range(Nch)
            ]
            HTB_vals, BLF_vals, BHF_vals, H_sig_vals, H_mc_vals, t_bhi_vals = zip(*out)
            HeartToBrain = np.stack(HTB_vals, axis=0)
            BrainToLF = np.stack(BLF_vals, axis=0)
            BrainToHF = np.stack(BHF_vals, axis=0)
            HeartToBrain_sigma = np.stack(H_sig_vals, axis=0)
            HeartToBrain_mc = np.stack(H_mc_vals, axis=0)
            time_bhi = t_bhi_vals[0]

        return HeartToBrain, BrainToLF, BrainToHF, time_bhi, HeartToBrain_sigma, HeartToBrain_mc

    def _compute_cpr_and_csr(self, win_RR, interpmethod=None):
        """
        This computes the time-varying parasympathetic and sympathetic indices CPr and CSr
        Input:
            - self:                 PySDG class instance  which has run compute_tfr_eeg
            - TFR_EEG:              numpy.array [Nc, Nt] - time-frequency response of the eeg signal for a specific frequency band
            - TFR_HRV:              numpy.array [Nt,] - time-frequency response of the hrv signal for a specific frequency band
            - win_RR:               window length in number of RR samples for the computation of CPr and CSr
            - interpmethod:         string (optional) specifies what internal interpolation method should be used
        Output:
            - CPr:                  numpy.array [Nt,] - time-varying parasympathetic index
            - CSr:                  numpy.array [Nt,] - time-varying sympathetic index


        """
        if interpmethod is None:
            interpmethod = interp1cubic
        RR = self.hrv["data"]

        # %% RR model parameter estimation
        # IPFM centre frequencies: the two sinusoids the model assumes modulate
        # heart rate. Set to the power-weighted spectral centroids measured in
        # this cohort (148 children, 5-10 yr) rather than the adult defaults of
        # 0.10 / 0.25 Hz:
        #   LF centroid  median 0.083 Hz  (P10-P90 0.076-0.092, no age effect)
        #   HF centroid  median 0.312 Hz  (peak 0.317; declines with age)
        # Note these are only weakly identified: CS/CP are min-max normalised
        # below, which removes most of the scaling, so swapping between 0.10/0.25,
        # 0.13/0.34 and 0.083/0.32 changes the normalised series by r >= 0.89
        # (CS) and r >= 0.97 (CP). Chosen for defensibility, not for effect.
        omega_lf = 2 * np.pi * 0.083  # LF centre frequency (pySDG default 0.10)
        omega_hf = 2 * np.pi * 0.32  # HF centre frequency (pySDG default 0.25)
        rr_cum = np.cumsum(RR)

        # Index_old is subtracted by 1 and index_new increased by one to accomodate for python 0 index and open-range slicing
        index_old = 0
        _w = np.where(rr_cum > 1 + win_RR)[0]
        index_new = _w[0] if len(_w) else index_old
        # NaN rather than 0, so a second that is never estimated below stays
        # missing instead of reading as a real zero-amplitude estimate.
        CS = np.full(
            [
                int(rr_cum[-1] - win_RR),
            ],
            np.nan,
        )
        CP = np.full(
            [
                int(rr_cum[-1] - win_RR),
            ],
            np.nan,
        )

        # For every second calculate a window to enable the calculation (overlapping segments)
        for i in range(0, int(rr_cum[-1] - win_RR)):
            # A window needs >= 2 RR intervals: L reads RR[old:new] and W reads
            # the successive differences RR[old+1:new] - RR[old:new-1]. Fewer
            # than that means the R-peak series has a gap longer than win_RR,
            # and upstream this raised on np.max of an empty slice. Leave the
            # second as NaN instead -- it is simply not modelled.
            if index_new - index_old < 2:
                index_old = np.where(rr_cum > (i + 1))[0][0]
                _w = np.where(rr_cum > (i + 1 + win_RR))[0]
                if not len(_w):
                    break
                index_new = _w[0]
                continue
            HR = 1 / np.mean(RR[(index_old + 1) : index_new])  # Time-varying heart rate
            gamma = np.sin(omega_hf / (2 * HR)) - np.sin(
                omega_lf / (2 * HR)
            )  # gamma parameter of the IPFM model
            MM = np.array(
                [
                    [
                        np.sin(omega_hf / (2 * HR))
                        * omega_lf
                        * HR
                        / (4 * np.sin(omega_lf / (2 * HR))),
                        -np.sqrt(2) * omega_lf * HR / (8 * np.sin(omega_lf / (2 * HR))),
                    ],
                    [
                        -np.sin(omega_lf / (2 * HR))
                        * omega_hf
                        * HR
                        / (4 * np.sin(omega_hf / (2 * HR))),
                        np.sqrt(2) * omega_hf * HR / (8 * np.sin(omega_hf / (2 * HR))),
                    ],
                ]
            )
            L = np.max(RR[index_old:index_new]) - np.min(RR[index_old:index_new])
            W = np.sqrt(2) * np.max(
                np.abs(RR[(index_old + 1) : index_new] - RR[index_old : (index_new - 1)])
            )
            CC = 1 / gamma * np.dot(MM, np.array([L, W]))
            CS[i] = CC[0]
            CP[i] = CC[1]
            index_old = np.where(rr_cum > (i + 1))[0][0]
            # The final iteration can ask for a beat past the end of the RR
            # series: int() floors the loop bound, so when rr_cum[-1] lands on
            # an exact integer the strict `>` has no match. Upstream this raised
            # IndexError; stop and leave the remaining seconds NaN instead.
            _w = np.where(rr_cum > (i + 1 + win_RR))[0]
            if not len(_w):
                break
            index_new = _w[0]

        # Interpolation and normalization to the [0, 1]-range of the parameters for computational reasons
        # rr_cum is relative to the FIRST R peak, not the movie onset.
        # Restore that origin before interpolation onto the EEG/movie clock.
        first_r_peak = self.hrv["t"][0] - RR[0]
        time_cs = first_r_peak + np.median(np.arange(1, win_RR + 1)) + np.arange(1, len(CS) + 1)

        CSr = interpmethod(time_cs, CS, self.tfr["t"])
        CPr = interpmethod(time_cs, CP, self.tfr["t"])

        CSr = (CSr - np.nanmin(CSr)) / (np.nanmax(CSr) - np.nanmin(CSr))
        CPr = (CPr - np.nanmin(CPr)) / (np.nanmax(CPr) - np.nanmin(CPr))

        return CPr, CSr

    def _BHI_InsideModel(self, TFR_ch, TFR_rr, CPr, CSr, time_tfr, TV):
        """
        Input:
            - TFR_ch:       numpy.array [Nt,] - time-frequency response of the eeg signal for a specific channel
            - TFR_rr:       numpy.array [Nt,] - time-frequency response of the hrv signal for a specific frequency band
            - CPr:          numpy.array [Nt,] - time-varying parasympathetic index
            - CSr:          numpy.array [Nt,] - time-varying sympathetic index
            - time_tfr:     numpy.array [Nt,] - timepoints corresponding to TFR_ch and TFR_rr
            - TV:           boolean flag whether time-varying BHI should be computed
        Output:
            - HToB:         numpy.array [Nt1,] - The computed instantaneous heart to-brain coupling for the specified eeg channel
            - BToLF:        numpy.array [Nt2,] - The computed instantaneous brain-to-heart coupling for the specified eeg channel (LF)
            - BToHF:        numpy.array [Nt2,] - The computed instantaneous brain-to-heart coupling for the specified eeg channel (HF)
            - HToB_sigma:   numpy.array [Nt1,] - Model parameter for fitting evaluation. This is the standard deviation of the residuals when fitting the ARX model.
            - HToB_mc:      numpy.array [Nt1,] - Model parameter for fitting evaluation. This is the AR coefficient of the ARX model. This values is effectively not used
            - time_bhi:     dict with numpy.arrays - Time indices corresponding to the coupling indices
                - time_bhi["time_HToB"]:    numpy.array - timepoints corresponding to HToB
                - time_bhi["time_BToH"]:    numpy.array - timepoints corresponding to BToLF and BToHF
        Description:

        """
        Nt = len(TFR_ch)
        Cs1 = self.opts[
            "Cs"
        ]  # The constant sympathetic term for subtraction prior to division by the historical eeg median
        Cp1 = self.opts[
            "Cp"
        ]  # The constant parasympathetic term for subtraction prior to division by the historical eeg median

        if TV:
            # The maximum number of iterations is calculated from the data lengths.
            Niterations = int(
                np.min([len(CPr), Nt - self.opts["wind"], len(TFR_rr) - self.opts["wind"]])
            )
            # Pre-allocation
            HToB_sigma = np.zeros(
                [
                    Niterations,
                ]
            )
            HToB_mc = np.zeros(
                [
                    Niterations,
                ]
            )
            HToB = np.zeros(
                [
                    Niterations,
                ]
            )
            time_HToB = np.zeros(
                [
                    Niterations,
                ]
            )
            medianTime_P_eeg = np.zeros(
                [
                    Niterations,
                ]
            )
            BToHF = np.zeros(
                [
                    Niterations,
                ]
            )
            BToLF = np.zeros(
                [
                    Niterations,
                ]
            )
            for i in range(0, Niterations):
                # ARX model estimation for every time window
                _y = TFR_ch[i : (i + self.opts["wind"] + 1)]
                _u = TFR_rr[i : (i + self.opts["wind"] + 1)]
                if np.isnan(_y).any() or np.isnan(_u).any():
                    # Window overlaps a gap in the R-peak series; an ARX fit on
                    # it is not defined. Mark, do not fit.
                    a1, x1, sigma2 = np.nan, np.nan, np.nan
                else:
                    a1, x1, sigma2 = self.opts["arx function"](
                        _y,
                        _u,
                        na=self.opts["arx orders"][0],
                        nb=self.opts["arx orders"][1],
                        nk=self.opts["arx orders"][2],
                    )
                HToB_sigma[i] = np.sqrt(sigma2)
                HToB_mc[i] = -a1
                HToB[i] = x1
                time_HToB[i] = time_tfr[int(self.opts["wind"] / 2 + i)]
                medianTime_P_eeg[i] = np.nanmedian(TFR_ch[i : (i + self.opts["wind"] + 1)])

                try:
                    # Assign simultaneous coupling to the window centre.
                    # Use cardiac parameters at the same nominal time as htb,
                    # the EEG window median, and the exported bth time axis.
                    center = i + self.opts["wind"] // 2
                    BToHF[i] = (CPr[center].squeeze() - Cp1) / medianTime_P_eeg[i]
                    BToLF[i] = (CSr[center].squeeze() - Cs1) / medianTime_P_eeg[i]
                except:
                    print("Exception has occured in BHI_InsideModel")

            # Same edge-only rule as the HRV TFR above. The leading ~8 s of
            # CPr/CSr is always NaN (interp1cubic has no support before the
            # first window mid-point), which is what makes bth shorter than
            # htb for every subject; that edge is still dropped. Interior NaN
            # from an R-peak gap is kept in place so bth stays on a uniform
            # time grid with the gap explicitly marked.
            _v = ~(np.isnan(BToLF) | np.isnan(BToHF))
            _lo, _hi = (int(_v.argmax()), len(_v) - int(_v[::-1].argmax())) if _v.any() else (0, 0)
            time_BToH = time_HToB[_lo:_hi]
            BToLF = BToLF[_lo:_hi]
            BToHF = BToHF[_lo:_hi]

            ## The np.convolve is not identical to matlabs conv, when using the same. Check the custom made convolve_name function for details
            # HToB =  np.convolve(HToB,  2/window*np.ones(int(window/2)), "same")
            # BToHF = np.convolve(BToHF, 2/window*np.ones(int(window/2)), "same")
            # BToLF = np.convolve(BToLF, 2/window*np.ones(int(window/2)), "same")
            HToB = convolve_same(HToB, 2 / self.opts["wind"] * np.ones(int(self.opts["wind"] / 2)))
            BToHF = convolve_same(
                BToHF, 2 / self.opts["wind"] * np.ones(int(self.opts["wind"] / 2))
            )
            BToLF = convolve_same(
                BToLF, 2 / self.opts["wind"] * np.ones(int(self.opts["wind"] / 2))
            )
        else:
            a1, x1, sigma2 = arx_least_squares(
                TFR_ch, TFR_rr, na=1, nb=1, nk=1
            )  # indices for the estimated parameters

            HToB_sigma = np.sqrt(sigma2)
            HToB_mc = a1
            HToB = x1
            time_HToB = 1
            medianTime_P_eeg = np.nanmedian(TFR_ch)

            # Brain to HF
            # The TFR_ch is normalized to [0,1], meaning that dividing by the individual values, like in the matlab code, will result in division by 0 somewhere along the array
            # Instead I chose to divide by the median of the array in line with what is done in the loop if TV==True
            BToHF = np.nanmedian((CPr - Cp1) / medianTime_P_eeg)
            BToLF = np.nanmedian((CSr - Cs1) / medianTime_P_eeg)
            time_BToH = 1

        time_bhi = dict()
        time_bhi["time_HToB"] = time_HToB
        time_bhi["time_BToH"] = time_BToH

        return [HToB, BToLF, BToHF, HToB_sigma, HToB_mc, time_bhi]

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ supporting functions ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
