function compute_eeg_band_power(id, CFG)
%% compute_eeg_band_power.m — Time-frequency power on continuous clean EEG
%
% Steps:
%   T01     Load continuous_clean.mat
%   T02     Split into 3 movie trials (fear / neutral / happy)
%   T03     ERPLAB epoch cleaning (2 s epochs): artifact detection -> per-epoch
%           spherical interp if salvageable, else NaN; stitched back to continuous
%   T04-T05 Per condition, cut to a common CFG.tf.analysisSec window:
%           ft_freqanalysis (mtmconvol, Hanning, 2 s window, 1 s step), then mean
%           band power over delta/theta/alpha/beta with NaN mask on bad time points
%   T06     Save {id}_movie_EEG_TS.mat
%
% Cleaning policy (continuous, preserves time alignment for BHI):
%   bad span <= cap channels -> spherical interp those channels in that span
%   bad span >  cap channels -> NaN all channels; downstream BHI skips that window
%
% The three movies differ in duration (~187 / ~190 / ~239 s). Cleaning runs on
% each condition's full length; the power step then cuts every condition to the
% first CFG.tf.analysisSec seconds, so all three share one time axis. time and
% nan_mask are still stored per condition (nan_mask differs even when time does).
%
% Input:   CFG.preprocessedOut/{id}_continuous_clean.mat
% Output:  CFG.eegTSOut/{id}_movie_EEG_TS.mat
%          Fields: movie_EEG_TS.{fear/neutral/happy}.{delta/theta/alpha/beta}
%                  (nchan x ntime, NaN where artifact; ntime = analysisSec - 1)
%                  movie_EEG_TS.time.{cond}, .nan_mask.{cond}, .freq, .chanlocs
%          qc_tf struct

%% T01  INIT + LOAD
if isnumeric(id), id = num2str(id, '%d'); end   % accept numeric IDs too
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'config'));
if nargin < 2, CFG = preprocessing_config(); end

% Each MATLAB batch process needs the plugin subfolders, even if eeglab is on path.
addpath(genpath(CFG.eeglabPath));
eeglab nogui
% FieldTrip: add root only (ft_defaults adds its own subfolders; never genpath)
if ~exist('ft_defaults','file'), addpath(CFG.fieldtripPath); end
ft_defaults;

if ~exist(CFG.eegTSOut,'dir'), mkdir(CFG.eegTSOut); end

fprintf('\n==========  TF  |  %s  ==========\n', id)

inFile = fullfile(CFG.preprocessedOut, sprintf('%s_continuous_clean.mat', id));
assert(exist(inFile,'file') > 0, 'continuous_clean not found: %s', inFile)
load(inFile, 'EEG','EEG_FT','seg_info','chanlocs0')

condNames = {'fear','neutral','happy'};
bandNames = {'delta','theta','alpha','beta'};
srate     = EEG.srate;
fprintf('\n[T01] loaded  |  %d chan, %d samp, %g Hz  (%.1f min)\n', ...
    EEG.nbchan, EEG.pnts, srate, EEG.pnts/srate/60)

%% T02  SEGMENT INTO 3 MOVIE TRIALS
FT            = EEG_FT;
FT.trial      = cell(1,3);
FT.time       = cell(1,3);
FT.sampleinfo = zeros(3,2);
for i = 1:3
    rng             = seg_info.onsets_new(i) : seg_info.offsets_new(i);
    FT.trial{i}     = EEG_FT.trial{1}(:, rng);
    FT.time{i}      = (0:numel(rng)-1) / srate;
    FT.sampleinfo(i,:) = [seg_info.onsets_new(i), seg_info.offsets_new(i)];
end
segLenSec = cellfun(@(t) numel(t), FT.time) / srate;
fprintf('\n[T02] segment lengths (s): %s\n', mat2str(round(segLenSec)))

%% T03  EPOCH-BASED CLEANING (ERPLAB) -> NaN-free continuous + bad-sample mask
% Per condition: cut into non-overlapping 2 s epochs, run ERPLAB artifact
% detection, then per epoch: spherical interp bad channels if <= cap. Epochs
% over cap (and the < 1 epoch tail) keep their raw data but are flagged in
% badSmp, which masks the POWER afterwards -- ft_freqanalysis cannot take NaN.
absThresh = CFG.eeg.artifactAbs;
ppThresh  = CFG.eeg.artifactPP;
ppWinMs   = CFG.eeg.artifactWin(1);
ppStepMs  = CFG.eeg.artifactWin(2);
maxBadCh  = CFG.eeg.maxBadChPerEpoch;
halfWin   = round(CFG.tf.winLen/2 * srate);
epL       = round(CFG.tf.cleanEpochSec * srate);   % epoch length (samples)
epMs      = CFG.tf.cleanEpochSec * 1000;

% Single-bin binlist (BDF): all regular events (code 1) -> bin 1
bdfFile = [tempname '.txt'];   % unique per process (concurrent-safe: fixed name clobbers across parallel workers/processes)
fid = fopen(bdfFile,'w'); fprintf(fid, 'bin 1\nepoch\n.{1}\n'); fclose(fid);

condBadSmp = cell(1,3);
fprintf('\n[T03] ERPLAB epoch cleaning (%g s, cap = %d ch):\n', ...
    CFG.tf.cleanEpochSec, maxBadCh)
fprintf(['[summary] Parameters: artifact epochs %g s; amplitude +/- %g uV; ' ...
    'peak-to-peak %g uV (%g/%g ms); interpolation cap %d channels.\n'], ...
    CFG.tf.cleanEpochSec, absThresh, ppThresh, ppWinMs, ppStepMs, maxBadCh)
fprintf(['[summary] Parameters: band power %s/%s; window %g s, step 1 s; first %g s; ' ...
    'bands delta %s, theta %s, alpha %s, beta %s Hz.\n'], ...
    CFG.tf.method, CFG.tf.taper, CFG.tf.winLen, CFG.tf.analysisSec, ...
    mat2str(CFG.bands.delta), mat2str(CFG.bands.theta), mat2str(CFG.bands.alpha), mat2str(CFG.bands.beta))
for ci = 1:3
    % Condition continuous data as an EEGLAB dataset
    cEEG  = pop_select(EEG, 'point', ...
        [seg_info.onsets_new(ci) seg_info.offsets_new(ci)]);
    nPtsC = cEEG.pnts;
    cont  = cEEG.data;                 % NaN-free continuous (raw), nchan x nPtsC

    % Insert regular time-locking events (code 1) at each epoch start
    starts     = 1 : epL : (nPtsC - epL + 1);
    cEEG.event = [];
    for k = 1:numel(starts)
        cEEG.event(k).type     = '1';
        cEEG.event(k).latency  = starts(k);
        cEEG.event(k).duration = 0;
    end
    cEEG = eeg_checkset(cEEG, 'eventconsistency');

    % ERPLAB: eventlist -> bin -> epoch (baseline 'none' keeps values intact)
    cEEG = pop_creabasiceventlist(cEEG, 'AlphanumericCleaning','on', ...
        'BoundaryString',{'boundary'}, 'BoundaryNumeric',{-99});
    cEEG = pop_binlister(cEEG, 'BDF',bdfFile, 'IndexEL',1, ...
        'SendEL2','EEG', 'Voutput','EEG');
    cEEG = pop_epochbin(cEEG, [0 epMs], 'none');

    % ERPLAB artifact detection: absolute value + moving-window peak-to-peak
    cEEG = pop_artextval(cEEG, 'Channel',1:cEEG.nbchan, 'Flag',1, ...
        'Threshold',[-absThresh absThresh], 'Twindow',[0 epMs]);
    cEEG = pop_artmwppth(cEEG, 'Channel',1:cEEG.nbchan, 'Flag',3, ...
        'Threshold',ppThresh, 'Twindow',[0 epMs], ...
        'Windowsize',ppWinMs, 'Windowstep',ppStepMs);

    % Per epoch: interp bad chans if <= cap (write into cont); else flag as bad
    nEp     = cEEG.trials;
    badSmp  = false(1, nPtsC);
    nInterp = 0;  nBad = 0;
    for e = 1:nEp
        cols  = (e-1)*epL + (1:epL);
        badCh = find(cEEG.reject.rejmanualE(:, e));
        if isempty(badCh), continue, end
        if numel(badCh) > maxBadCh
            badSmp(cols) = true;   nBad = nBad + 1;   % keep raw data, mask later
        else
            epi = pop_selectevent(cEEG, 'epoch', e, ...
                'deleteevents','off', 'deleteepochs','on', 'invertepochs','off');
            epi = eeg_interp(epi, badCh, 'spherical');
            cont(:, cols) = epi.data;   nInterp = nInterp + 1;
        end
    end
    badSmp(nEp*epL+1 : end) = true;       % tail (< 1 epoch) -> masked
    FT.trial{ci}   = cont;                % NaN-free, fed to ft_freqanalysis
    condBadSmp{ci} = badSmp;

    fprintf('       %-7s : %d ep -> interp %d | bad %d | tail %d smp\n', ...
        condNames{ci}, nEp, nInterp, nBad, nPtsC - nEp*epL)
    fprintf(['[summary] %s %s: %d epochs; %d with channel interpolation; ' ...
        '%d bad; %d tail samples masked.\n'], ...
        id, condNames{ci}, nEp, nInterp, nBad, nPtsC - nEp*epL)
end
delete(bdfFile);

%% T04-T05  FREQANALYSIS + BAND POWER  (per condition, common 178 s window)
% Movies differ in duration (~187 / ~190 / ~239 s), so each one is cut to the
% first CFG.tf.analysisSec seconds and the three then share a single time axis.
% Band power is the MEAN PSD over the band (independent of band width and
% frequency resolution, so a band gaining or losing a bin does not rescale it).
movie_EEG_TS = struct();
nTimeArr = zeros(1,3);
nRejArr  = zeros(1,3);

fprintf('\n[T04-T05] freqanalysis + band power (mtmconvol Hanning, mean):\n')
for ci = 1:3
    cond = condNames{ci};

    % Single-trial FieldTrip struct for this condition, cut to the common window
    nKeep = round(CFG.tf.analysisSec * srate);
    assert(size(FT.trial{ci}, 2) >= nKeep, ...
        '%s / %s: segment is %.1f s, shorter than the %g s analysis window', ...
        id, cond, size(FT.trial{ci},2)/srate, CFG.tf.analysisSec)

    FTc            = EEG_FT;
    FTc.trial      = {FT.trial{ci}(:, 1:nKeep)};
    FTc.time       = {FT.time{ci}(1:nKeep)};
    FTc.sampleinfo = FT.sampleinfo(ci,1) + [0, nKeep-1];

    toi   = 1 : 1 : CFG.tf.analysisSec - 1;   % 1 Hz time points (s)
    nTime = numel(toi);

    cfg            = [];
    cfg.output     = 'pow';
    cfg.method     = CFG.tf.method;
    cfg.taper      = CFG.tf.taper;
    cfg.foi        = CFG.tf.foi;
    cfg.t_ftimwin  = ones(numel(cfg.foi),1) * CFG.tf.winLen;
    cfg.toi        = toi;
    cfg.keeptrials = 'yes';
    TFR = ft_freqanalysis(cfg, FTc);   % 1 x nchan x nfreq x nTime

    % toi windows overlapping any flagged (artifact/tail) sample -> masked
    nanSmp   = condBadSmp{ci};
    nan_mask = false(1, nTime);
    for ti = 1:nTime
        center = round(toi(ti) * srate);
        loT = max(1, center - halfWin + 1);
        hiT = min(numel(nanSmp), center + halfWin);
        nan_mask(ti) = any(nanSmp(loT:hiT));
    end

    % ft_freqanalysis snaps foi onto the trial's DFT grid, so TFR.freq can sit
    % ~1e-3 Hz off the requested values. Comparing the raw axis against the band
    % edges then silently drops (or keeps) an edge bin depending on trial length,
    % which made the bands differ between subjects. Round back onto the requested
    % grid first so the selection is exactly the nominal one, every time.
    fStep = CFG.tf.foi(2) - CFG.tf.foi(1);
    fNom  = round(TFR.freq / fStep) * fStep;
    assert(max(abs(fNom - TFR.freq)) < fStep/4, ...
        '%s / %s: TFR.freq is not a perturbation of CFG.tf.foi', id, cond)

    for bi = 1:numel(bandNames)
        band = bandNames{bi};
        fIdx = fNom >= CFG.bands.(band)(1) & fNom <= CFG.bands.(band)(2);
        pwr  = squeeze(mean(TFR.powspctrm(1, :, fIdx, :), 3));  % nchan x nTime
        pwr(:, nan_mask) = NaN;
        movie_EEG_TS.(cond).(band) = pwr;
    end
    movie_EEG_TS.time.(cond)     = toi;
    movie_EEG_TS.nan_mask.(cond) = nan_mask;

    nTimeArr(ci) = nTime;
    nRejArr(ci)  = sum(nan_mask);
    fprintf('       %-7s : %.1f s -> %g s | ntime %d | %d masked\n', ...
        cond, segLenSec(ci), CFG.tf.analysisSec, nTime, nRejArr(ci))
    fprintf('[summary] %s %s power: %d time points; %d masked for artifacts/tail.\n', id, cond, nTime, nRejArr(ci))
end
movie_EEG_TS.freq     = TFR.freq;
movie_EEG_TS.chanlocs = chanlocs0;

%% T06  SAVE
qc_tf = struct( ...
    'id',              id, ...
    'n_time_fear',     nTimeArr(1), ...
    'n_time_neutral',  nTimeArr(2), ...
    'n_time_happy',    nTimeArr(3), ...
    'n_rej_fear',      nRejArr(1), ...
    'n_rej_neutral',   nRejArr(2), ...
    'n_rej_happy',     nRejArr(3), ...
    'pct_rej_fear',    100*nRejArr(1)/nTimeArr(1), ...
    'pct_rej_neutral', 100*nRejArr(2)/nTimeArr(2), ...
    'pct_rej_happy',   100*nRejArr(3)/nTimeArr(3), ...
    'params_tf',       CFG.tf, ...
    'params_bands',    CFG.bands, ...
    'artifact_abs_uv', absThresh, ...
    'artifact_pp_uv',  ppThresh, ...
    'artifact_win_ms', CFG.eeg.artifactWin, ...
    'max_bad_ch',      maxBadCh, ...
    'created',         datestr(now,'yyyy-mm-dd HH:MM:SS'));

outFile = fullfile(CFG.eegTSOut, sprintf('%s_movie_EEG_TS.mat', id));
save(outFile, 'movie_EEG_TS','qc_tf', '-v7.3')
fprintf('\n[T06] saved → %s\n', outFile)
fprintf('[summary] Saved: %s\n', outFile)
fprintf('\n==========  TF DONE  |  rej f=%d n=%d h=%d  ==========\n\n', ...
    nRejArr(1), nRejArr(2), nRejArr(3))
end
