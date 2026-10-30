function compute_hep_roi_waveforms(hepDir, outFile, infoFile)
%% compute_hep_roi_waveforms.m — per-subject ROI-mean HEP waveforms from {id}_hep_epochs.mat
%
% Average retained trials per condition, then the ROI channels in CFG.roi.
%
%   compute_hep_roi_waveforms(CFG.hepEcgRegOut, 'roi_waves_ecgreg.mat', 'participants.csv')
% Optional infoFile restricts the files to its ID column, retaining filename order.
%
% Output (-v7, readable with scipy.io.loadmat): times (1xT), ids (cellstr),
% n_trials (n x 3, fear/neutral/happy), and <Region>__<cond> (n x T) arrays.

here = fileparts(mfilename('fullpath'));
addpath(here); addpath(fullfile(here, '..', '..', '01_preprocessing', 'config'));
CFG = preprocessing_config();
if nargin < 1, hepDir = fullfile(here, '..', 'outputs', 'batch', 'hep_epochs_ecgreg'); end
if nargin < 2, outFile = fullfile(here, '..', 'outputs', 'batch', 'averages', 'roi_waves_ecgreg.mat'); end
conds = {'fear', 'neutral', 'happy'};
regs  = fieldnames(CFG.roi);

files = dir(fullfile(hepDir, '*_hep_epochs.mat'));
if nargin >= 3 && ~isempty(infoFile)
    opts = detectImportOptions(infoFile, 'TextType', 'string');
    assert(ismember('ID', opts.VariableNames), 'Participant CSV needs an ID column.');
    opts = setvartype(opts, 'ID', 'string');
    info = readtable(infoFile, opts);
    requested = strtrim(string(info.ID));
    assert(~any(ismissing(requested) | strlength(requested) == 0) && ...
        numel(unique(requested)) == numel(requested), ...
        'Participant IDs must be unique and nonempty.');
    available = erase(string({files.name}), '_hep_epochs.mat');
    missing = requested(~ismember(requested, available));
    assert(isempty(missing), 'Missing HEP epoch files for: %s', strjoin(missing, ', '));
    files = files(ismember(available, requested));
end
assert(~isempty(files), 'No HEP epochs in %s; run extract_hep_epochs first.', hepDir);
outDir = fileparts(outFile);
if ~isempty(outDir) && ~isfolder(outDir), mkdir(outDir); end
ids = cell(numel(files), 1);
W = struct();
n_trials = zeros(numel(files), 3);
times = [];
for k = 1:numel(files)
    S = load(fullfile(files(k).folder, files(k).name), 'EEG', 'conditions');
    ids{k} = erase(files(k).name, '_hep_epochs.mat');
    if isempty(times)
        times = S.EEG.times;
        for r = 1:numel(regs), for c = 1:3
            W.([regs{r} '__' conds{c}]) = nan(numel(files), numel(times));
        end, end
    end
    cl = lower(string(S.conditions(:)));
    for c = 1:3
        sel = cl == conds{c};
        n_trials(k, c) = sum(sel);
        if ~any(sel), continue; end
        m = mean(double(S.EEG.data(:, :, sel)), 3);      % chan x time
        for r = 1:numel(regs)
            W.([regs{r} '__' conds{c}])(k, :) = mean(m(CFG.roi.(regs{r}), :), 1);
        end
    end
    if mod(k, 10) == 0, fprintf('%d / %d\n', k, numel(files)); end
end
W.times = times; W.ids = ids; W.n_trials = n_trials;
save(outFile, '-struct', 'W', '-v7');
fprintf('-> %s  (%d subjects)\n', outFile, numel(files));
end
