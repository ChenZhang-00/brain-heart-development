function CFG = preprocessing_config()
%% preprocessing_config.m — central configuration for EEG preprocessing pipeline
% Paths are anchored to the repository. Analysis parameters follow the manuscript.

%% PATHS
% Resolve the repository root from this file, independent of the working directory.
projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

% Raw BrainVision files: rawDataPath/{ID}_movie.vhdr (+ .eeg / .vmrk)
CFG.rawDataPath = fullfile(projectRoot, '00_data', 'raw_data');
if ~isempty(getenv('DBH_RAW_DATA_PATH'))
    CFG.rawDataPath = getenv('DBH_RAW_DATA_PATH');
end

% Each processing stage owns its generated arrays. Single-subject runs write to
% outputs/demo; the notebook sets DBH_PREPROCESSING_OUTPUT for a CSV batch
% (outputs/batch). The manuscript arrays in outputs/manuscript are read only.
preOut = fullfile(projectRoot, '01_preprocessing', 'outputs', 'demo');
if ~isempty(getenv('DBH_PREPROCESSING_OUTPUT'))
    preOut = getenv('DBH_PREPROCESSING_OUTPUT');
end
hepOut = fullfile(projectRoot, '04_HEP', 'outputs', 'demo');
CFG.dataOut             = preOut;
CFG.correctedRpeakPath  = fullfile(preOut, 'ecg', 'r_peaks');
CFG.preprocessedOut    = fullfile(preOut, 'eeg', 'cleaned');
CFG.eegTSOut           = fullfile(preOut, 'eeg', 'band_power');

% ECG-regressed HEP preparation.
CFG.ecgTrimmedPath        = fullfile(hepOut, 'ecg_trimmed');
CFG.preprocessedEcgRegOut = fullfile(hepOut, 'preprocessed_ecgreg');
CFG.hepEcgRegOut          = fullfile(hepOut, 'hep_epochs_ecgreg');
CFG.ecgreg.estHp          = 1;   % Hz — high-pass used only to estimate the regression coefficients

% EEGLAB with BVA-io, DIPFIT, Zapline-plus, CleanLine, ICLabel and ERPLAB.
CFG.eeglabPath = fullfile(projectRoot, 'toolboxes', 'eeglab');

% FieldTrip root (used by compute_eeg_band_power.m for ft_freqanalysis)
CFG.fieldtripPath = fullfile(projectRoot, 'toolboxes', 'fieldtrip');

% The guided notebook can point to existing local toolbox installations.
if ~isempty(getenv('DBH_EEGLAB_PATH'))
    CFG.eeglabPath = getenv('DBH_EEGLAB_PATH');
end
if ~isempty(getenv('DBH_FIELDTRIP_PATH'))
    CFG.fieldtripPath = getenv('DBH_FIELDTRIP_PATH');
end

%% EEG PREPROCESSING (preprocess_eeg.m)

% High-pass: two values because ICA needs stricter HP than the saved output
CFG.eeg.hpMain          = 0.1;   % Hz — saved output (supports HEP down to 0.1 Hz)
CFG.eeg.hpICA           = 1.0;   % Hz — used only for ICA fitting
CFG.eeg.lp              = 45;    % Hz — low-pass (common upper bound for both)
CFG.eeg.filterOrder     = 4;     % Butterworth order

% Bad channel detection (on movie-trimmed, 1 Hz HP data)
CFG.eeg.badChanSdThresh  = 200;  % µV — reject channel if SD across time > this
CFG.eeg.maxGlobalBadChan = 8;    % hard cap on channels removed before ICA

% ICLabel component rejection (applied to Muscle/Eye/Heart/LineNoise/ChanNoise)
CFG.eeg.icLabelThresh = 0.8;     % reject IC if any non-brain class probability > this

% Artifact detection (used in downstream scripts, stored here for traceability)
CFG.eeg.artifactAbs      = 150;     % µV — absolute amplitude threshold
CFG.eeg.artifactPP       = 150;     % µV — moving-window peak-to-peak threshold
CFG.eeg.artifactWin      = [200 50]; % ms — [window width, step size]
CFG.eeg.maxBadChPerEpoch = 13;       % per-epoch/span spherical interp cap (~20% of 63)

%% HEP EPOCHING (extract_hep_epochs.m) — per methods + shared with CFG.eeg
CFG.hep.bandpass         = [0.1 30];   % Hz — low-pass to 30 (0.1 HP already applied upstream)
CFG.hep.bpOrder          = 4;          % Butterworth order
CFG.hep.epoch            = [-0.2 0.6]; % s — R-locked epoch window (-200 to 600 ms)
CFG.hep.baselineMs       = [-200 0];   % ms — baseline correction window
CFG.hep.artifactAbs      = CFG.eeg.artifactAbs;      % uV
CFG.hep.ppThresh         = CFG.eeg.artifactPP;       % uV
CFG.hep.ppWindowMs       = CFG.eeg.artifactWin(1);   % ms
CFG.hep.ppStepMs         = CFG.eeg.artifactWin(2);   % ms
CFG.hep.maxBadChPerEpoch = CFG.eeg.maxBadChPerEpoch; % per-epoch interp cap
CFG.hep.doPlot           = false;      % no QC plot in batch

%% TF / BAND PARAMETERS (compute_eeg_band_power.m)

CFG.tf.method  = 'mtmconvol';
CFG.tf.taper   = 'hanning';
CFG.tf.foi     = 1:0.5:45;   % Hz — frequency axis for STFT
CFG.tf.winLen  = 2;          % s  — spectrogram window length (50% overlap → 1 s step)
CFG.tf.cleanEpochSec = 2;    % s  — epoch length for ERPLAB artifact cleaning
CFG.tf.analysisSec   = 178;  % s  — common analysis window: every condition is cut
                             %      to this length so the three share one time axis.
                             %      178 s at 500 Hz also puts the whole foi grid
                             %      (k * 0.5 Hz) exactly on the trial's DFT bins.

% Frequency band edges [lo hi] Hz
CFG.bands.delta = [1   4];
CFG.bands.theta = [4   7];
CFG.bands.alpha = [7  12];
CFG.bands.beta  = [12 30];

%% ROI CHANNEL INDICES (63-ch layout, after dropping ch64 ECG + ch65 EDA)

CFG.roi.Frontal   = [1,32,33,34,35,62,63,5,4,37,3,36,2,61,31,60,30,29];
CFG.roi.Central   = [38,6,39,7,27,58,28,59,9,41,8,40,26,57,25,56,24,42,10,43,11,53,22,54,23,55];
CFG.roi.Posterior = [15,14,45,13,44,12,52,21,51,20,19,46,47,48,49,50,16,17,18];

end
