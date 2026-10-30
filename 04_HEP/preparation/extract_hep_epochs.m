function extract_hep_epochs(id, CFG)
% ECG-regressed HEP epochs: -200 to 600 ms, baseline -200 to 0 ms.
% Interpolate up to 13 bad channels per epoch; discard epochs exceeding this cap.
% Input: CFG.preprocessedEcgRegOut/{id}_continuous_clean_ecg_regressed.mat
%        and corrected R peaks from CFG.correctedRpeakPath.
% Output: CFG.hepEcgRegOut/{id}_hep_epochs.mat.

%% H01  INIT + LOAD
if isnumeric(id), id = num2str(id, '%d'); end   % accept numeric IDs too
if isstring(id),  id = char(id);          end   % ... and string scalars

% Locate the shared configuration and the NPZ reader.
here = fileparts(mfilename('fullpath'));        % '' if pasted into the console
if ~isempty(here)
    addpath(here);
    for c = {fullfile(here,'..','..','01_preprocessing','config'), here}
        if exist(fullfile(c{1},'preprocessing_config.m'),'file'), addpath(c{1}); break; end
    end
end
assert(exist('preprocessing_config','file') > 0, ...
    'preprocessing_config.m not on the path — add 01_preprocessing/config first.')
if nargin < 2, CFG = preprocessing_config(); end
CFG.hepOut = CFG.hepEcgRegOut;

% EEGLAB: always add root + all subfolders. exist('eeglab','file') is not a
% safe guard — the root can be on the path while the plugins (which live in
% subfolders: ERPLAB -> pop_basicfilter, dipfit -> eeglab2fieldtrip) are not,
% and that only surfaces later as "Undefined function 'pop_basicfilter'".
% Re-genpath whenever any function we need is missing or resolves to a
% *different* install (e.g. a stray ERPLAB9.10 or eeglab2023.1 on the saved path).
assert(exist(CFG.eeglabPath,'dir') > 0, 'EEGLAB not found: %s', CFG.eeglabPath)
needFcn = {'eeglab','pop_basicfilter','eeg_interp','pop_eegthresh', ...
           'pop_epoch','pop_rmbase','pop_select','eeglab2fieldtrip'};
pathOK = true;
for f = needFcn
    w = which(f{1});
    if isempty(w) || ~strncmpi(w, CFG.eeglabPath, numel(CFG.eeglabPath))
        pathOK = false; break
    end
end
if ~pathOK
    wState = warning('off','MATLAB:dispatcher:nameConflict');
    addpath(genpath(CFG.eeglabPath));
    % ICLabel bundles a matconvnet gather.m that shadows the MATLAB builtin.
    shadow = fullfile(CFG.eeglabPath,'plugins','ICLabel','matconvnet', ...
                      'matlab','compatibility','parallel');
    if exist(shadow,'dir'), rmpath(shadow); end
    warning(wState);
    eeglab nogui
    for f = needFcn
        assert(exist(f{1},'file') > 0, ...
            '%s still missing after addpath(genpath(''%s'')) — check the EEGLAB install.', ...
            f{1}, CFG.eeglabPath)
    end
end

if ~exist(CFG.hepOut,'dir'), mkdir(CFG.hepOut); end

fprintf('\n==========  HEP  |  %s  |  %s  ==========\n', ...
    id, datestr(now,'yyyy-mm-dd HH:MM:SS'))
tStart = tic;
fprintf('[summary] Parameters: HEP low-pass %g Hz; epoch [%g %g] s; baseline [%g %g] ms; amplitude %g uV; peak-to-peak %g uV; at most %d bad channels per epoch\n', ...
    CFG.hep.bandpass(2), CFG.hep.epoch, CFG.hep.baselineMs, ...
    CFG.hep.artifactAbs, CFG.hep.ppThresh, CFG.hep.maxBadChPerEpoch)

inFile = fullfile(CFG.preprocessedEcgRegOut, sprintf('%s_continuous_clean_ecg_regressed.mat', id));
assert(exist(inFile,'file') > 0, 'ECG-regressed EEG not found: %s', inFile)
load(inFile, 'EEG','seg_info','chanlocs0')

% Drop the ICA solution carried over from preprocess_eeg. It was already *applied*
% there (P09 pop_subcomp), so the weights are stale against this data. Leaving
% them in makes every eeg_checkset / pop_select call in H05-H07 recompute an
% 85 MB icaact matrix, which then gets saved into each output file (~1/3 of its
% size). The solution itself is still recorded in {id}_continuous_clean.mat.
[EEG.icaact, EEG.icaweights, EEG.icasphere, EEG.icawinv, EEG.icachansind] = deal([]);

condNames = {'fear','neutral','happy'};
srate     = EEG.srate;
fprintf('\n[H01] loaded  |  %d chan, %d samp, %g Hz  (%.1f min)\n', ...
    EEG.nbchan, EEG.pnts, srate, EEG.pnts/srate/60)

%% H02  30 Hz LOW-PASS
fprintf('\n[H02] LP %g Hz ... ', CFG.hep.bandpass(2)); ts = tic;
EEG = pop_basicfilter(EEG, 1:EEG.nbchan, ...
    'Boundary','boundary', 'Cutoff', CFG.hep.bandpass(2), ...
    'Design','butter', 'Filter','lowpass', 'Order', CFG.hep.bpOrder);
fprintf('done (%.1fs)\n', toc(ts))

%% H03  LOAD CORRECTED R-PEAKS, REMAP TO TRIMMED TIMELINE
fprintf('\n[H03] loading R-peaks ... ')
npzPath = fullfile(CFG.correctedRpeakPath, sprintf('%s_movie_rpeaks.npz', id));
assert(exist(npzPath,'file') > 0, '.npz not found: %s', npzPath)

% Pure-MATLAB .npz read (read_npz.m, alongside this file): no py.numpy.load,
% no pyenv, no temp directory — the same call works on any machine and inside
% parfor workers.
npz        = read_npz(npzPath);
rPeaksOrig = round(double(npz.r_peaks(:)));

% Guard against a rate/recording mismatch, which would silently misalign every
% epoch: the .npz records the ECG sampling rate and raw length it was cut from.
if isfield(npz,'fs')
    assert(abs(double(npz.fs) - srate) < 1e-6, ...
        'sampling-rate mismatch: R-peaks at %g Hz but EEG at %g Hz', ...
        double(npz.fs), srate)
end
if isfield(npz,'n_samples')
    assert(seg_info.offsets_orig(3) <= double(npz.n_samples), ...
        'seg_info runs past the ECG recording (%d > %d samples)', ...
        seg_info.offsets_orig(3), double(npz.n_samples))
end

% Keep only peaks inside movie segments; remap to trimmed sample indices
rPeaksTrim = [];
cumOff     = 0;
for i = 1:3
    inSeg      = rPeaksOrig >= seg_info.onsets_orig(i) & ...
                 rPeaksOrig <= seg_info.offsets_orig(i);
    rPeaksTrim = [rPeaksTrim; ...
        rPeaksOrig(inSeg) - seg_info.onsets_orig(i) + 1 + cumOff]; %#ok<AGROW>
    cumOff     = cumOff + seg_info.offsets_orig(i) - seg_info.onsets_orig(i) + 1;
end
fprintf('total %d  |  in-movie %d\n', numel(rPeaksOrig), numel(rPeaksTrim))
% Note: R-peaks were already visually verified and manually corrected in the
% R-peak editor, so no automatic IRI-based rejection is applied here. HEP
% morphology (incl. any QRS residual) is checked in H07b.

%% H04  CONDITION + SAFE ZONE
epWinSmp = round(CFG.hep.epoch * srate);
preBuf   = -epWinSmp(1);
postBuf  =  epWinSmp(2);

inSafe  = false(size(rPeaksTrim));
condIdx = zeros(size(rPeaksTrim));
for i = 1:3
    lo = seg_info.onsets_new(i)  + preBuf;
    hi = seg_info.offsets_new(i) - postBuf;
    in = rPeaksTrim >= lo & rPeaksTrim <= hi;
    inSafe(in)  = true;
    condIdx(in) = i;
end
rPeaksUse    = rPeaksTrim(inSafe);
useCond      = condIdx(inSafe);
conditions   = condNames(useCond)';
tInMovie_sec = (rPeaksUse - seg_info.onsets_new(useCond)') / srate;

fprintf('\n[H04] safe-zone R: %d  (fear=%d  neutral=%d  happy=%d)\n', ...
    numel(rPeaksUse), sum(useCond==1), sum(useCond==2), sum(useCond==3))

%% H05  EPOCH
fprintf('\n[H05] epoching [%g %g] s ... ', CFG.hep.epoch); ts = tic;
rEventTypes = strcat('R_', condNames);

if isstruct(EEG.event) && ~isempty(EEG.event)
    EEG.event = EEG.event(~ismember({EEG.event.type}, [{'R'} rEventTypes]));
else
    EEG.event = struct('type',{},'latency',{},'duration',{});   % continuous_clean stores no events
end
if isempty(EEG.event)
    template = struct('type','','latency',0,'duration',0);
else
    template = structfun(@(~)[], EEG.event(1), 'UniformOutput',false);
end
newEv = repmat(template, 1, numel(rPeaksUse));
for k = 1:numel(rPeaksUse)
    newEv(k).type    = rEventTypes{useCond(k)};
    newEv(k).latency  = rPeaksUse(k);
    newEv(k).duration = 0;
end
EEG.event = [EEG.event(:)', newEv];
EEG = eeg_checkset(EEG, 'eventconsistency');
EEG = pop_epoch(EEG, rEventTypes, CFG.hep.epoch);
assert(EEG.trials == numel(rPeaksUse), ...
    'pop_epoch dropped events: in=%d out=%d', numel(rPeaksUse), EEG.trials)
fprintf('done (%.1fs)  |  %d trials  fear=%d  neutral=%d  happy=%d\n', ...
    toc(ts), EEG.trials, sum(useCond==1), sum(useCond==2), sum(useCond==3))

%% H06  BASELINE
EEG = pop_rmbase(EEG, CFG.hep.baselineMs);
fprintf('\n[H06] baseline [%d %d] ms\n', CFG.hep.baselineMs)

%% H07  CLEAN — abs + p-p → per-epoch spherical interp → drop
fprintf('\n[H07] artifact rejection\n')

% Absolute amplitude
EEG  = pop_eegthresh(EEG, 1, 1:EEG.nbchan, ...
    -CFG.hep.artifactAbs, CFG.hep.artifactAbs, ...
    CFG.hep.epoch(1), CFG.hep.epoch(2), 0, 0);
nAbs = sum(EEG.reject.rejthresh);
fprintf('       abs ±%d µV        : %d / %d\n', CFG.hep.artifactAbs, nAbs, EEG.trials)

% Moving-window peak-to-peak
ppWin   = round(CFG.hep.ppWindowMs/1000 * srate);
ppStep  = max(1, round(CFG.hep.ppStepMs/1000 * srate));
ppFlagE = false(EEG.nbchan, EEG.trials);
for k = 1:EEG.trials
    pp = movmax(EEG.data(:,:,k), ppWin, 2) - movmin(EEG.data(:,:,k), ppWin, 2);
    ppFlagE(:,k) = any(pp(:, 1:ppStep:end) > CFG.hep.ppThresh, 2);
end
EEG.reject.rejthreshE = EEG.reject.rejthreshE | ppFlagE;
EEG.reject.rejthresh  = EEG.reject.rejthresh  | any(ppFlagE,1);
nPP = sum(any(ppFlagE,1));
fprintf('       p-p >%d µV (%dms) : %d / %d\n', CFG.hep.ppThresh, CFG.hep.ppWindowMs, nPP, EEG.trials)

% Per-epoch spherical interp (rescues epochs with few bad channels)
cleanData = EEG.data;
nInterp = 0;  nDrop = 0;
for k = 1:EEG.trials
    badMask = EEG.reject.rejthreshE(:,k);
    nBad    = sum(badMask);
    if nBad == 0, continue; end
    if nBad > CFG.hep.maxBadChPerEpoch, nDrop = nDrop + 1; continue; end
    EEGk = eeg_interp(pop_select(EEG,'trial',k), find(badMask));
    cleanData(:,:,k)             = EEGk.data;
    EEG.reject.rejthresh(k)      = false;
    EEG.reject.rejthreshE(:,k)   = false;
    nInterp = nInterp + 1;
end
EEG.data = cleanData;
fprintf('       interp %d  |  drop-flagged %d (>%d bad ch)\n', ...
    nInterp, nDrop, CFG.hep.maxBadChPerEpoch)

% Drop remaining flagged trials
keepIdx      = find(~EEG.reject.rejthresh);
nBefore      = EEG.trials;
assert(~isempty(keepIdx), ...
    'all %d epochs rejected for %s — check CFG.hep artifact thresholds', nBefore, id)
EEG          = pop_select(EEG, 'trial', keepIdx);
conditions   = conditions(keepIdx);
tInMovie_sec = tInMovie_sec(keepIdx);
fprintf('       kept %d / %d  (%.1f%%)\n', EEG.trials, nBefore, 100*EEG.trials/nBefore)

%% H07b  QC PLOT — HEP butterfly (optional)
if CFG.hep.doPlot
    fprintf('\n[H07b] HEP QC plot ... ')
    condColors = [0.85 0.20 0.20; 0.30 0.30 0.30; 0.20 0.50 0.85];
    tEp = EEG.times;

    chLbls = lower({EEG.chanlocs.labels});
    cc = 1;
    for q = {'cz','fcz','fz','cpz','pz'}
        h = find(strcmp(chLbls, q{1}), 1);
        if ~isempty(h), cc = h; break; end
    end

    figure('Color','w','Position',[100 80 1500 720])
    for c = 1:3
        isCond = strcmp(conditions, condNames{c});
        subplot(2,3,c)
        if ~any(isCond), title(sprintf('%s N=0',condNames{c})); axis off; continue; end
        erp = mean(EEG.data(:,:,isCond),3);
        plot(tEp, erp','Color',[.7 .7 .7]); hold on
        plot(tEp, mean(erp,1),'k','LineWidth',1.6)
        xline(0,'k--'); yline(0,'k:')
        xlabel('ms post-R'); ylabel('µV'); grid on; axis tight
        title(sprintf('%s  N=%d', condNames{c}, sum(isCond)), ...
            'Color',condColors(c,:),'FontWeight','bold')
    end
    subplot(2,3,4:6); hold on
    for c = 1:3
        isCond = strcmp(conditions, condNames{c});
        if ~any(isCond), continue; end
        plot(tEp, mean(EEG.data(cc,:,isCond),3), ...
            'Color',condColors(c,:),'LineWidth',1.8, ...
            'DisplayName', sprintf('%s (N=%d)', condNames{c}, sum(isCond)));
    end
    xline(0,'k--'); yline(0,'k:')
    xlabel('ms post-R'); ylabel('µV'); grid on; axis tight
    title(sprintf('Channel %s — 3 conditions', EEG.chanlocs(cc).labels))
    legend('Location','best')
    fprintf('done\n')
end

%% H08  SAVE
fprintf('\n[H08] saving ... '); ts = tic;
EEG_FT = eeglab2fieldtrip(EEG, 'preprocessing', 'none');

qc_hep = struct( ...
    'id',                   id, ...
    'n_R_loaded',           numel(rPeaksOrig), ...
    'n_R_in_movie',         numel(rPeaksTrim), ...
    'n_R_safe_zone',        numel(rPeaksUse), ...
    'n_kept_total',         EEG.trials, ...
    'n_kept_fear',          sum(strcmp(conditions,'fear')), ...
    'n_kept_neutral',       sum(strcmp(conditions,'neutral')), ...
    'n_kept_happy',         sum(strcmp(conditions,'happy')), ...
    'n_interp',             nInterp, ...
    'n_dropped',            numel(rPeaksUse) - EEG.trials, ...
    'params',               CFG.hep, ...
    'elapsed_s',            toc(tStart), ...
    'created',              datestr(now,'yyyy-mm-dd HH:MM:SS'));

outFile = fullfile(CFG.hepOut, sprintf('%s_hep_epochs.mat', id));
save(outFile, 'EEG','EEG_FT','conditions','tInMovie_sec','qc_hep', ...
    'chanlocs0','seg_info', '-v7.3')

fprintf('done (%.1fs)\n  → %s\n', toc(ts), outFile)
fprintf('[summary] %s HEP: kept %d/%d epochs (fear %d, neutral %d, happy %d); interpolated %d epochs; rejected %d\n', ...
    id, EEG.trials, nBefore, qc_hep.n_kept_fear, qc_hep.n_kept_neutral, ...
    qc_hep.n_kept_happy, nInterp, nBefore - EEG.trials)
fprintf('\n==========  HEP DONE  |  %.1f min  |  kept %d (f=%d n=%d h=%d)  ==========\n\n', ...
    qc_hep.elapsed_s/60, qc_hep.n_kept_total, ...
    qc_hep.n_kept_fear, qc_hep.n_kept_neutral, qc_hep.n_kept_happy)
end
