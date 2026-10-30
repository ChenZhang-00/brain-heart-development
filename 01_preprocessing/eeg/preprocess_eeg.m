function preprocess_eeg(id)
%% preprocess_eeg.m — Core EEG cleaning for the movie-viewing task
%
% Steps:
%   P01  Load BrainVision .vhdr
%   P02  Remove non-EEG channels (ch64 ECG + ch65 EDA)
%   P03  Line noise: Zapline-plus → CleanLine
%   P04  Bandpass: two versions — 0.1 Hz HP (saved output) & 1 Hz HP (ICA only)
%   P05  Find S1–S6 movie markers
%   P06  Trim to 3 movie segments, concatenate
%   P07  Global bad channel detection on 1 Hz version (SD threshold)
%   P08  ICA on 1 Hz HP trimmed data (runica, extended)
%   P09  ICLabel → flag non-brain ICs → apply solution to 0.1 Hz data
%   P10  Restore removed channels via spherical interpolation
%   P11  Average re-reference
%   P12  Save {id}_continuous_clean.mat
%
% Input:   CFG.rawDataPath/{id}_movie.vhdr
% Output:  CFG.preprocessedOut/{id}_continuous_clean.mat
%          Fields: EEG, EEG_FT, seg_info, chanlocs0
%          (ECG is not stored here — it lives in CFG.dataOut/ecg/extracted/ and CFG.dataOut/ecg/r_peaks/)
%
% Downstream:
%   extract_hep_epochs.m  — R-locked HEP epochs (applies 30 Hz LP on top)
%   compute_eeg_band_power.m     — STFT power time series (artifact mask + NaN)
%
% Required plugins: BVA-io, DIPFIT, Zapline-plus, CleanLine, ICLabel, ERPLAB.

%% INIT
if isnumeric(id), id = num2str(id, '%d'); end   % accept numeric IDs too
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'config'));
CFG = preprocessing_config();

if ~exist('eeglab','file'), addpath(CFG.eeglabPath); end
eeglab nogui

addpath(genpath(CFG.eeglabPath));

if ~exist(CFG.preprocessedOut,'dir'), mkdir(CFG.preprocessedOut); end

fprintf('\n==========  EEG PRE  |  %s  ==========\n', id)
fprintf(['[summary] Parameters: EEG main %g-%g Hz; ICA %g-%g Hz; Butterworth order %d; ' ...
    'bad-channel SD > %g uV, cap %d; ICLabel %g.\n'], ...
    CFG.eeg.hpMain, CFG.eeg.lp, CFG.eeg.hpICA, CFG.eeg.lp, CFG.eeg.filterOrder, ...
    CFG.eeg.badChanSdThresh, CFG.eeg.maxGlobalBadChan, CFG.eeg.icLabelThresh)

%% P01  LOAD
% raw_data is flat: raw_data/{id}_movie.vhdr (+ .eeg / .vmrk)
fprintf('\n[P01] loading ... ')
vhdrName = sprintf('%s_movie.vhdr', id);
inFile   = fullfile(CFG.rawDataPath, vhdrName);
assert(exist(inFile,'file') > 0, 'File not found: %s', inFile)
EEG = pop_loadbv(CFG.rawDataPath, vhdrName, [], []);
fprintf('done  |  %d chan, %d samp, %g Hz\n', EEG.nbchan, EEG.pnts, EEG.srate)
fprintf('[summary] %s EEG input: %d channels, %d samples, %g Hz.\n', id, EEG.nbchan, EEG.pnts, EEG.srate)

%% P02  REMOVE NON-EEG CHANNELS
% Channel 64 = ECG, channel 65 = EDA. ECG is handled by the separate ECG
% pipeline (extract_ecg.py → R-peak editor), so it is dropped here.
EEG        = pop_select(EEG, 'nochannel', [64 65]);
chanlocs0  = EEG.chanlocs;        % 63-channel reference layout for later interp
nbchan_ref = EEG.nbchan;          % = 63
fprintf('[P02] ch64 ECG + ch65 EDA removed  |  %d EEG channels\n', EEG.nbchan)

%% P03  LINE NOISE
fprintf('\n[P03] line noise ... ')

EEG = pop_zapline_plus(EEG, 'noisefreqs','line', ...
    'coarseFreqDetectPowerDiff',4, 'chunkLength',0, ...
    'adaptiveNremove',1, 'fixedNremove',1, 'plotResults',0);

EEG = pop_cleanline(EEG, 'bandwidth',2, 'chanlist',1:EEG.nbchan, ...
    'computepower',1, 'linefreqs',50, 'newversion',0, 'normSpectrum',0, ...
    'p',0.01, 'pad',2, 'plotfigures',0, 'scanforlines',1, ...
    'sigtype','Channels', 'tau',100, 'verb',0, ...
    'winsize',4, 'winstep',1);
fprintf('done\n')

%% P04  BANDPASS — main (0.1 Hz HP) and ICA-fitting (1 Hz HP)
fprintf('\n[P04] bandpass ... ')

EEG_main = pop_basicfilter(EEG, 1:EEG.nbchan, ...
    'Boundary','boundary', ...
    'Cutoff', [CFG.eeg.hpMain CFG.eeg.lp], ...
    'Design','butter', 'Filter','bandpass', 'Order', CFG.eeg.filterOrder);

EEG_ica = pop_basicfilter(EEG, 1:EEG.nbchan, ...
    'Boundary','boundary', ...
    'Cutoff', [CFG.eeg.hpICA CFG.eeg.lp], ...
    'Design','butter', 'Filter','bandpass', 'Order', CFG.eeg.filterOrder);

fprintf('done  |  main [%.1f–%g Hz]  ICA [%.1f–%g Hz]\n', ...
    CFG.eeg.hpMain, CFG.eeg.lp, CFG.eeg.hpICA, CFG.eeg.lp)

%% P05  FIND MOVIE MARKERS (S1–S6)
% Restarts can repeat markers. Select the last onset/offset pair spanning
% 150–260 s, rather than an aborted take.
fprintf('\n[P05] locating S1–S6 ... ')
evTypes = {EEG.event.type};
evLats  = [EEG.event.latency];

CLIP_MIN = 150;   CLIP_MAX = 260;   % s — the three clips are ~187 / 190 / 238.5
S = zeros(1,6);
for c = 1:3
    a = 2*c - 1;   b = 2*c;
    on  = evLats(strcmp(evTypes, sprintf('S  %d', a)));
    off = evLats(strcmp(evTypes, sprintf('S  %d', b)));
    assert(~isempty(on) && ~isempty(off), ...
        'Marker S%d or S%d not found for subject %s', a, b, id)

    dur  = (off(:)' - on(:)) / EEG.srate;      % numel(on) x numel(off), in s
    good = find(dur >= CLIP_MIN & dur <= CLIP_MAX);
    assert(~isempty(good), ...
        '%s: no S%d/S%d pair with a plausible duration (candidates %s s)', ...
        id, a, b, mat2str(dur(:)', 4))

    [ii, jj]  = ind2sub(size(dur), good);
    [~, pick] = max(on(ii));                   % the LAST valid take
    S(a) = on(ii(pick));   S(b) = off(jj(pick));

    if numel(on) > 1 || numel(off) > 1
        fprintf(['\n       S%d/S%d repeated (%d/%d markers, %d valid pair(s)) ' ...
                 '-> using %d-%d (%.1f s)\n       '], ...
            a, b, numel(on), numel(off), numel(good), ...
            S(a), S(b), (S(b) - S(a)) / EEG.srate)
    end
end
fprintf('S1=%d  S2=%d  S3=%d  S4=%d  S5=%d  S6=%d\n', S(1),S(2),S(3),S(4),S(5),S(6))

% Condition order: fear (S1–S2), neutral (S3–S4), happy (S5–S6)
onsets_orig  = [S(1), S(3), S(5)];
offsets_orig = [S(2), S(4), S(6)];

%% P06  TRIM TO MOVIE SEGMENTS → CONCATENATE
fprintf('\n[P06] trimming to movie ... ')
segLens     = offsets_orig - onsets_orig + 1;
onsets_new  = [1, segLens(1)+1, segLens(1)+segLens(2)+1];
offsets_new = onsets_new + segLens - 1;
nTotal      = sum(segLens);

seg_info = struct( ...
    'onsets_orig',  onsets_orig, ...
    'offsets_orig', offsets_orig, ...
    'onsets_new',   onsets_new, ...
    'offsets_new',  offsets_new, ...
    'cond_order',   {{'fear','neutral','happy'}});

trimIdx = [];
for i = 1:3
    trimIdx = [trimIdx, onsets_orig(i):offsets_orig(i)]; %#ok<AGROW>
end

% Build trimmed versions by copying structure and replacing data
EEG_main_m        = EEG_main;
EEG_main_m.data   = EEG_main.data(:, trimIdx);
EEG_main_m.pnts   = nTotal;
EEG_main_m.xmax   = (nTotal - 1) / EEG_main_m.srate;
EEG_main_m.times  = (0:nTotal-1) / EEG_main_m.srate * 1000;
EEG_main_m.event  = [];
EEG_main_m        = eeg_checkset(EEG_main_m);

EEG_ica_m         = EEG_ica;
EEG_ica_m.data    = EEG_ica.data(:, trimIdx);
EEG_ica_m.pnts    = nTotal;
EEG_ica_m.xmax    = EEG_main_m.xmax;
EEG_ica_m.times   = EEG_main_m.times;
EEG_ica_m.event   = [];
EEG_ica_m         = eeg_checkset(EEG_ica_m);

fprintf('done  |  %d samp (%.1f min per condition)\n', nTotal, mean(segLens)/EEG_main_m.srate/60)
fprintf('[summary] %s movie durations (fear/neutral/happy): %s s.\n', id, mat2str(segLens / EEG_main_m.srate, 5))

%% P07  GLOBAL BAD CHANNEL DETECTION (on 1 Hz HP trimmed data)
fprintf('\n[P07] bad channel detection (SD > %g µV) ... ', CFG.eeg.badChanSdThresh)
sdPerChan   = std(EEG_ica_m.data, 0, 2);
badChanMask = sdPerChan > CFG.eeg.badChanSdThresh;
badChanIdx  = find(badChanMask);

if numel(badChanIdx) > CFG.eeg.maxGlobalBadChan
    [~, ord] = sort(sdPerChan, 'descend');
    badChanIdx  = ord(1:CFG.eeg.maxGlobalBadChan);
    badChanMask = false(EEG_ica_m.nbchan, 1);
    badChanMask(badChanIdx) = true;
end

if ~isempty(badChanIdx)
    badChanNames = {EEG_ica_m.chanlocs(badChanIdx).labels};
    EEG_main_m = pop_select(EEG_main_m, 'nochannel', badChanIdx);
    EEG_ica_m  = pop_select(EEG_ica_m,  'nochannel', badChanIdx);
    fprintf('removed %d: %s\n', numel(badChanIdx), strjoin(badChanNames, ', '))
else
    fprintf('none\n')
end
fprintf('[summary] %s bad EEG channels removed before ICA: %d [%s].\n', ...
    id, numel(badChanIdx), strjoin({chanlocs0(badChanIdx).labels}, ', '))

%% P08  ICA
fprintf('\n[P08] ICA (runica, extended) on %d channels × %d samp ... ', ...
    EEG_ica_m.nbchan, EEG_ica_m.pnts);
EEG_ica_m = pop_runica(EEG_ica_m, 'icatype','runica', 'extended',1, ...
    'stop',1e-7, 'interupt','off');
fprintf('done\n')

%% P09  ICLabel — classify on main data, remove flagged ICs
fprintf('\n[P09] ICLabel (threshold = %.2f) ... ', CFG.eeg.icLabelThresh)

% Transfer ICA solution from 1 Hz version to 0.1 Hz version
EEG_main_m.icaweights = EEG_ica_m.icaweights;
EEG_main_m.icasphere  = EEG_ica_m.icasphere;
EEG_main_m            = eeg_checkset(EEG_main_m);   % computes icaact

% Classify components
EEG_main_m = pop_iclabel(EEG_main_m, 'default');

% Reject: Muscle / Eye / Heart / Line Noise / Channel Noise > threshold
% ICLabel columns: [Brain Muscle Eye Heart LineNoise ChanNoise Other]
thr = CFG.eeg.icLabelThresh;
EEG_main_m = pop_icflag(EEG_main_m, ...
    [NaN NaN; thr 1; thr 1; thr 1; thr 1; thr 1; NaN NaN]);

badIC = find(EEG_main_m.reject.gcompreject);
EEG_main_m = pop_subcomp(EEG_main_m, badIC, 0);
fprintf('removed %d ICs  [%s]\n', numel(badIC), num2str(badIC(:)'))
fprintf('[summary] %s ICA components removed: %d [%s].\n', id, numel(badIC), num2str(badIC(:)'))

%% P10  RESTORE GLOBALLY REMOVED CHANNELS (spherical interpolation)
fprintf('\n[P10] restoring %d bad channel(s) ... ', numel(badChanIdx))
if ~isempty(badChanIdx)
    nPts = EEG_main_m.pnts;
    tmp  = zeros(nbchan_ref, nPts, 'single');
    goodMask = ~ismember(1:nbchan_ref, badChanIdx);
    tmp(goodMask, :)       = EEG_main_m.data;
    EEG_main_m.data        = tmp;
    EEG_main_m.chanlocs    = chanlocs0;
    EEG_main_m.nbchan      = nbchan_ref;
    EEG_main_m             = eeg_checkset(EEG_main_m);
    EEG_main_m             = eeg_interp(EEG_main_m, badChanIdx, 'spherical');
end
fprintf('done  |  %d channels\n', EEG_main_m.nbchan)

%% P11  AVERAGE RE-REFERENCE
EEG_main_m = pop_reref(EEG_main_m, []);
fprintf('\n[P11] average re-referenced\n')
fprintf(['[summary] %s EEG output: %d channels, %d samples; ' ...
    '%d channels interpolated; average reference.\n'], ...
    id, EEG_main_m.nbchan, EEG_main_m.pnts, numel(badChanIdx))

%% P12  SAVE
fprintf('\n[P12] saving ... ')
EEG    = EEG_main_m;
EEG_FT = eeglab2fieldtrip(EEG, 'preprocessing', 'none');

outFile = fullfile(CFG.preprocessedOut, sprintf('%s_continuous_clean.mat', id));
save(outFile, 'EEG','EEG_FT','seg_info','chanlocs0', '-v7.3')

fprintf('done\n  → %s\n', outFile)
fprintf('[summary] Saved: %s\n', outFile)
fprintf('\n==========  DONE  |  %d IC removed  |  %d chan interp  ==========\n\n', ...
    numel(badIC), numel(badChanIdx))
end
