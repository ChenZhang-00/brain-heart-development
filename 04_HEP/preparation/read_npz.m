function S = read_npz(npzPath, varName)
%READ_NPZ  Read a NumPy .npz archive in pure MATLAB — no Python, no pyenv.
%
%   S    = read_npz(file)            struct with every array in the archive
%   arr  = read_npz(file, 'r_peaks') just that array
%
% Parses the ZIP container and the .npy headers directly, so stored members
% (the default for np.savez) never touch the filesystem: no temp directory,
% no unzip, no cleanup — safe to call from parfor workers. Deflated members
% (np.savez_compressed) fall back to MATLAB's built-in unzip.
%
% Supported dtypes: int8/16/32/64, uint8/16/32/64, float32/64, bool, and
% unicode ('<U..', returned as char / cellstr). C and Fortran order.
%
% Written for the HEP pipeline (extract_hep_epochs.m) to replace py.numpy.load,
% which required a matching CPython install on every machine.

narginchk(1, 2)
assert(exist(npzPath, 'file') > 0, 'read_npz: file not found: %s', npzPath)

members = unzipInMemory(npzPath);

S = struct();
for k = 1:numel(members)
    [~, base] = fileparts(members(k).name);
    S.(matlab.lang.makeValidName(base)) = parseNPY(members(k).bytes, members(k).name);
end

if nargin > 1
    varName = char(varName);
    [~, varName] = fileparts(varName);          % tolerate 'r_peaks.npy'
    fld = matlab.lang.makeValidName(varName);
    assert(isfield(S, fld), 'read_npz: "%s" not in %s (has: %s)', ...
        varName, npzPath, strjoin(fieldnames(S)', ', '))
    S = S.(fld);
end
end


% ---------------------------------------------------------------------------
function members = unzipInMemory(zipPath)
% Minimal ZIP reader. Returns struct array with .name and .bytes (uint8).

fid = fopen(zipPath, 'r', 'l');
assert(fid > 0, 'read_npz: cannot open %s', zipPath)
raw = onCleanupRead(fid);

n = numel(raw);
assert(n > 22, 'read_npz: %s is too small to be a .npz', zipPath)

% End-of-central-directory record: 'PK\5\6', within the last 64 kB + 22 B
from = max(1, n - 65557);
hit  = strfind(double(raw(from:n))', [80 75 5 6]);
assert(~isempty(hit), 'read_npz: no ZIP end-of-central-directory in %s', zipPath)
eocd = from - 1 + hit(end);

nEntries = double(typecast(raw(eocd+10 : eocd+11), 'uint16'));
cdStart  = double(typecast(raw(eocd+16 : eocd+19), 'uint32')) + 1;

members = repmat(struct('name', '', 'bytes', uint8([])), 1, nEntries);
needUnzip = false;
p = cdStart;
for k = 1:nEntries
    assert(isequal(double(raw(p:p+3))', [80 75 1 2]), ...
        'read_npz: bad central-directory entry %d in %s', k, zipPath)
    method   = double(typecast(raw(p+10 : p+11), 'uint16'));
    compSize = double(typecast(raw(p+20 : p+23), 'uint32'));
    fnLen    = double(typecast(raw(p+28 : p+29), 'uint16'));
    exLen    = double(typecast(raw(p+30 : p+31), 'uint16'));
    cmLen    = double(typecast(raw(p+32 : p+33), 'uint16'));
    lhOff    = double(typecast(raw(p+42 : p+45), 'uint32')) + 1;
    name     = char(raw(p+46 : p+45+fnLen))';

    members(k).name = name;
    if method == 0
        % Stored: the local header's own name/extra lengths give the payload.
        assert(isequal(double(raw(lhOff:lhOff+3))', [80 75 3 4]), ...
            'read_npz: bad local header for %s in %s', name, zipPath)
        lfnLen = double(typecast(raw(lhOff+26 : lhOff+27), 'uint16'));
        lexLen = double(typecast(raw(lhOff+28 : lhOff+29), 'uint16'));
        dStart = lhOff + 30 + lfnLen + lexLen;
        members(k).bytes = raw(dStart : dStart + compSize - 1);
    else
        needUnzip = true;                       % savez_compressed
    end
    p = p + 46 + fnLen + exLen + cmLen;
end

if needUnzip
    tmpd = tempname; mkdir(tmpd);
    cl = onCleanup(@() rmdir(tmpd, 's')); %#ok<NASGU>
    unzip(zipPath, tmpd);
    for k = 1:nEntries
        if isempty(members(k).bytes)
            f = fopen(fullfile(tmpd, members(k).name), 'r', 'l');
            assert(f > 0, 'read_npz: unzip did not produce %s', members(k).name)
            members(k).bytes = onCleanupRead(f);
        end
    end
end
end


% ---------------------------------------------------------------------------
function b = onCleanupRead(fid)
% fread the whole file as uint8 and always close the handle.
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
b = fread(fid, inf, '*uint8');
end


% ---------------------------------------------------------------------------
function a = parseNPY(b, name)
% Decode .npy bytes (format v1/v2/v3) into a MATLAB array.

assert(numel(b) > 10 && isequal(char(b(2:6))', 'NUMPY'), ...
    'read_npz: %s is not a .npy stream', name)
if b(7) >= 2
    hLen = double(typecast(b(9:12), 'uint32'));  hOff = 12;
else
    hLen = double(typecast(b(9:10), 'uint16'));  hOff = 10;
end
hdr  = char(b(hOff+1 : hOff+hLen))';
body = b(hOff+hLen+1 : end);

descr = regexp(hdr, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
assert(~isempty(descr), 'read_npz: %s has an unsupported dtype (structured?)', name)
descr = descr{1};
shape = regexp(hdr, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
dims  = sscanf(strrep(shape{1}, ',', ' '), '%f')';
fortran = ~isempty(regexp(hdr, '''fortran_order''\s*:\s*True', 'once'));

swap = descr(1) == '>';
kind = descr(2);
w    = str2double(descr(3:end));
if isnan(w), w = 1; end

switch kind
    case 'i', prec = sprintf('int%d',  w*8);
    case 'u', prec = sprintf('uint%d', w*8);
    case 'f'
        switch w
            case 4, prec = 'single';
            case 8, prec = 'double';
            otherwise, error('read_npz: unsupported float width in %s (%s)', name, descr)
        end
    case 'b', prec = 'uint8';
    case 'U', prec = 'uint32';                  % UTF-32LE, w codepoints/element
    case 'S', prec = 'uint8';                   % w bytes/element
    otherwise
        error('read_npz: unsupported dtype %s in %s', descr, name)
end

a = typecast(body, prec);
if swap, a = swapbytes(a); end

% Unicode / bytes: the fixed width is an extra leading dimension of chars.
if kind == 'U' || kind == 'S'
    c = reshape(char(a), w, []);
    c(c == 0) = ' ';                            % numpy zero-pads fixed widths
    a = cellstr(deblank(c'));
    if isempty(dims), a = a{1}; else, a = reshape(a, [dims 1]); end
    return
end
if kind == 'b', a = logical(a); end

if isempty(dims)
    a = a(1);                                   % 0-d array -> scalar
elseif isscalar(dims)
    a = reshape(a, dims, 1);
elseif fortran
    a = reshape(a, dims);
else
    a = permute(reshape(a, fliplr(dims)), numel(dims):-1:1);
end
end
