function regress_ecg_from_eeg(id, CFG)
%% regress_ecg_from_eeg.m — remove the cardiac field artefact by linear regression
%
% Sits between preprocess_eeg (continuous_clean) and extract_hep_epochs. For every
% condition segment and every channel it fits, on the whole segment,
%     EEG_ch(t) = b1 * ECG(t) + b2 * dECG/dt(t) + b0 + residual(t)
% and keeps the residual. The coefficients are ESTIMATED on 1 Hz high-passed
% copies of EEG and ECG (slow drift otherwise dominates the covariance and
% corrupts the fit) and then APPLIED to the original 0.1 Hz data. The QRS
% dominates the estimate, so a few-microvolt HEP barely influences it; the
% derivative term absorbs a small phase lag between lead II and the scalp.
%
% Input:   CFG.preprocessedOut/{id}_continuous_clean.mat      (preprocess_eeg)
%          CFG.ecgTrimmedPath/{id}_ecg_trimmed.npz            (export_aligned_ecg.py;
%          0.1-45 Hz ECG on the trimmed timeline, R up, same length as EEG.pnts)
% Output:  CFG.preprocessedEcgRegOut/{id}_continuous_clean_ecg_regressed.mat
%          same variables as the input (EEG, seg_info, chanlocs0; EEG_FT dropped)
%          plus qc_ecgreg: beta (chan x 3 x cond), r2 (chan x cond), qrs_ratio
%          (chan x cond: R-locked peak |amp| after / before, from all R-peaks)
%
% extract_hep_epochs(id) reads the ECG-regressed file.

if isnumeric(id), id = num2str(id, '%d'); end
if isstring(id),  id = char(id);          end

here = fileparts(mfilename('fullpath'));
if ~isempty(here)
    addpath(here)
    addpath(fullfile(here, '..', '..', '01_preprocessing', 'config'));
end
if nargin < 2, CFG = preprocessing_config(); end
if ~exist(CFG.preprocessedEcgRegOut, 'dir'), mkdir(CFG.preprocessedEcgRegOut); end

inFile  = fullfile(CFG.preprocessedOut, sprintf('%s_continuous_clean.mat', id));
ecgFile = fullfile(CFG.ecgTrimmedPath,  sprintf('%s_ecg_trimmed.npz', id));
rpkFile = fullfile(CFG.correctedRpeakPath, sprintf('%s_movie_rpeaks.npz', id));
assert(isfile(inFile),  'continuous_clean not found: %s', inFile)
assert(isfile(ecgFile), 'ecg_trimmed not found: %s', ecgFile)
fprintf('\n==========  ECG-REGRESS  |  %s  ==========\n', id)
t0 = tic;

load(inFile, 'EEG', 'seg_info', 'chanlocs0')
npz = read_npz(ecgFile);
ecg = double(npz.ecg(:))';
assert(numel(ecg) == EEG.pnts, '%s: ECG length %d ~= EEG.pnts %d', id, numel(ecg), EEG.pnts)
srate = EEG.srate;
X = double(EEG.data);                       % chan x pnts
decg = gradient(ecg) * srate;               % derivative, units/s

% 1 Hz high-passed copies for coefficient estimation only
[bh, ah] = butter(4, CFG.ecgreg.estHp / (srate/2), 'high');
Xh   = filtfilt(bh, ah, X')';
ecgh = filtfilt(bh, ah, ecg);
decgh = gradient(ecgh) * srate;

nCh   = EEG.nbchan;
beta  = zeros(nCh, 3, 3);
r2    = zeros(nCh, 3);
for c = 1:3
    idx = seg_info.onsets_new(c):seg_info.offsets_new(c);
    Rh  = [ecgh(idx)', decgh(idx)', ones(numel(idx), 1)]; % T x 3, estimation
    Yh  = Xh(:, idx)';
    B   = Rh \ Yh;                                        % 3 x chan, OLS on 1 Hz HP data
    fith = Rh * B;
    r2(:, c)     = 1 - sum((Yh - fith).^2, 1)' ./ sum((Yh - mean(Yh, 1)).^2, 1)';
    beta(:, :, c) = B';
    R   = [ecg(idx)', decg(idx)', ones(numel(idx), 1)];   % apply to 0.1 Hz data
    X(:, idx) = (X(:, idx)' - R * B)';
end

% QC: how much of the R-locked QRS is left (all in-movie R-peaks, +-100 ms)
qrs_ratio = nan(nCh, 3);
if isfile(rpkFile)
    rp = read_npz(rpkFile);
    rOrig = round(double(rp.r_peaks(:)));
    w = round(0.1 * srate);
    for c = 1:3
        r = rOrig(rOrig >= seg_info.onsets_orig(c) & rOrig <= seg_info.offsets_orig(c)) ...
            - seg_info.onsets_orig(c) + seg_info.onsets_new(c);
        r = r(r > w & r <= EEG.pnts - w);
        if numel(r) < 20, continue; end
        accB = zeros(nCh, 2*w+1); accA = accB;
        for k = 1:numel(r)
            accB = accB + double(EEG.data(:, r(k)-w:r(k)+w));
            accA = accA + X(:, r(k)-w:r(k)+w);
        end
        qrs_ratio(:, c) = max(abs(accA), [], 2) ./ max(abs(accB), [], 2);
    end
end

EEG.data = single(X);
EEG.setname = sprintf('%s_continuous_clean_ecg_regressed', id);
qc_ecgreg = struct('id', id, 'beta', beta, 'r2', r2, 'qrs_ratio', qrs_ratio, ...
    'regressors', {{'ecg', 'decg_dt', 'const'}}, 'ecg_band', [0.1 45], 'est_hp', CFG.ecgreg.estHp, ...
    'created', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

outFile = fullfile(CFG.preprocessedEcgRegOut, sprintf('%s_continuous_clean_ecg_regressed.mat', id));
save(outFile, 'EEG', 'seg_info', 'chanlocs0', 'qc_ecgreg', '-v7.3')
fprintf('[summary] %s ECG regression: %d channels, %d samples; ECG and derivative with intercept; coefficients estimated at %g Hz high-pass\n', ...
    id, EEG.nbchan, EEG.pnts, CFG.ecgreg.estHp)
fprintf('R2 median %.3f (max %.3f) | QRS residual ratio median %.2f | %.1fs\n  -> %s\n', ...
    median(r2(:)), max(r2(:)), median(qrs_ratio(:), 'omitnan'), toc(t0), outFile)
end
