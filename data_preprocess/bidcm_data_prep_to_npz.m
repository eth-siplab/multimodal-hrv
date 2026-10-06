%% BIDMC -> segs (8s windows, 6s overlap), IMU/TEMP as zeros, uses clean_ibi_ecg
clear; clc;

% --- Paths ---
basedir  = "";                           % where to save output
infile   = fullfile(basedir, "bidmc_data.mat");          % input .mat with variable 'data'
tooldir  = "";  % your funcs

addpath(tooldir);   % gives access to clean_ibi_ecg.m and pan_tompkin.m (if present)

% --- Output ---
out_file = fullfile(basedir, "segments_8s_hop2s_bidmc.mat");

S = load(infile);
assert(isfield(S,"data"), "Expected variable 'data' in %s", infile);
data = S.data;

% --- Config ---
fs_target = 128;             % unify to 128 Hz (8 s = 1024 samples)
win_s = 8;                   % window length (s)
hop_s = 2;                   % 6 s overlap
bp_lohi = [0.5 8];           % PPG bandpass (Hz)
ma_ms = 250;                 % moving avg on PPG
ibi_minmax_ms = [300 2000];  % ECG IBI range gate
temp_fs = 1;                 % dummy TEMP Fs (zeros)

% Precompute lengths
win_len = win_s * fs_target;     % 1024
hop_len = hop_s * fs_target;     % 256
ma_win  = round((ma_ms/1000) * fs_target);

% PPG filter (for resampled PPG)
[b,a] = butter(2, bp_lohi/(fs_target/2), "bandpass");

% Template (same fields as your other dataset)
template = struct("subject","", "fs",struct(), ...
                  "ecg",[], "ppg",[], "imu",[], "temp",[], "ibi_ecg_ms",[]);
segs(1:numel(data)) = template;   % preallocate

%% Iterate subjects
for i = 1:numel(data)
    % Raw + Fs
    ppg_raw = double(data(i).ppg.v);
    fs_ppg  = double(data(i).ppg.fs);
    ecg_raw = double(data(i).ekg.v);
    fs_ecg  = double(data(i).ekg.fs);

    % Resample to 128 Hz
    if fs_ppg ~= fs_target
        ppg = resample(ppg_raw, fs_target, fs_ppg);
    else
        ppg = ppg_raw;
    end
    if fs_ecg ~= fs_target
        ecg = resample(ecg_raw, fs_target, fs_ecg);
    else
        ecg = ecg_raw;
    end

    % PPG: bandpass + normalize + smooth
    ppg = filtfilt(b,a,ppg);
    ppg = (ppg - mean(ppg,'omitnan')) ./ std(ppg, 0, 'omitnan');
    if ma_win > 1
        ppg = movmean(ppg, ma_win, 'Endpoints','shrink');
    end

    % Usable duration
    tmax_s = min(numel(ppg)/fs_target, numel(ecg)/fs_target);
    if tmax_s < win_s
        warning("Subject %d shorter than 8 s. Skipping.", i);
        segs(i) = template; segs(i).subject = sprintf("subj_%02d", i);
        continue;
    end

    % Window starts (s) and count
    starts_s = 0:hop_s:(tmax_s - win_s);
    nW = numel(starts_s);

    % Prealloc
    ecg_win = zeros(nW, win_len);
    ppg_win = zeros(nW, win_len);
    imu_xM  = zeros(nW, win_len);
    imu_yM  = zeros(nW, win_len);
    imu_zM  = zeros(nW, win_len);
    tmp_len = win_s * temp_fs;   % 8 samples @ 1 Hz
    temp_w  = cell(nW,1);
    ibi_ecg_ms = cell(nW,1);

    % Windows
    for k = 1:nW
        s = floor(starts_s(k) * fs_target) + 1;
        e = s + win_len - 1;
        if e > numel(ecg) || e > numel(ppg)
            % guard
            ecg_win(k,:) = 0; ppg_win(k,:) = 0;
            imu_xM(k,:) = 0; imu_yM(k,:) = 0; imu_zM(k,:) = 0;
            temp_w{k} = zeros(tmp_len,1);
            ibi_ecg_ms{k} = [];
            continue;
        end

        ecg_seg = ecg(s:e);
        ppg_seg = ppg(s:e);

        ecg_win(k,:) = ecg_seg;
        ppg_win(k,:) = ppg_seg;

        % IMU zeros
        imu_xM(k,:) = 0; imu_yM(k,:) = 0; imu_zM(k,:) = 0;

        % TEMP zeros (keep same container style)
        temp_w{k} = zeros(tmp_len,1);

        % QRS -> IBIs (ms)
        try
            [~, qrs_i_raw, ~] = pan_tompkin(ecg_seg, fs_target, 0);   % requires pan_tompkin on path
        catch
            warning("pan_tompkin not found or failed for subject %d, window %d. IBIs empty.", i, k);
            qrs_i_raw = [];
        end

        figure,
        plot(ecg_seg)
        hold on
        plot(qrs_i_raw, ecg_seg(qrs_i_raw), '*r')

        if numel(qrs_i_raw) >= 2
            rr_ms = diff(qrs_i_raw) / fs_target * 1000;   % IBIs in ms

            % Use your cleaner (must be on path via addpath(tooldir))
            try
                % If your clean_ibi_ecg returns (rr_clean, keep_mask, qc_struct), adapt args
                [rr_clean, ~, ~] = clean_ibi_ecg(rr_ms);  % typical signature
            catch
                % Fallback: simple range-gate if clean_ibi_ecg signature differs
                rr_clean = rr_ms(isfinite(rr_ms) & rr_ms >= ibi_minmax_ms(1) & rr_ms <= ibi_minmax_ms(2));
            end
            ibi_ecg_ms{k} = rr_clean(:);
        else
            ibi_ecg_ms{k} = [];
        end
    end

    % Pack IMU
    imu_win = cat(3, imu_xM, imu_yM, imu_zM);  % [nW x 1024 x 3]

    % Subject ID
    if isfield(data(i),"name") && ~isempty(data(i).name)
        subj_id = char(string(data(i).name));
    elseif isfield(data(i),"rec") && ~isempty(data(i).rec)
        subj_id = char(string(data(i).rec));
    else
        subj_id = sprintf("subj_%02d", i);
    end

    % Build seg
    seg.subject  = subj_id;
    seg.fs = struct('ecg',fs_target,'ppg',fs_target,'imu',fs_target,'temp',temp_fs,...
                    'win_s',win_s,'hop_s',hop_s);
    seg.ecg  = ecg_win;                 % [nW x 1024]
    seg.ppg  = ppg_win;                 % [nW x 1024]
    seg.imu  = imu_win;                 % [nW x 1024 x 3] zeros
    seg.temp = cell2mat(temp_w);        % zeros
    seg.ibi_ecg_ms = ibi_ecg_ms;        % {nW x 1} cleaned IBIs per window

    segs(i) = seg;
end

% Save
save(out_file, "segs", "-v7.3");
fprintf("Saved %d subject segment structs to:\n  %s\n", numel(segs), out_file);
