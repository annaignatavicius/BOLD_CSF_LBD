%% CALUCLATE PARCELWISE BOLD–CSF COUPLING 

% Lag selection: for each parcel, choose lag from grand mean of group means (equal weight per group), same principle as gBOLD–CSF.

clear; clc;

%% Set parameters

TR         = 3.0;     % seconds
maxLagSec  = 18;      % +/- seconds
maxLagTR   = round(maxLagSec / TR);
nLags      = 2*maxLagTR + 1;

rootControls = '/path/to/controls';
rootDLB      = '/path/to/DLB';
rootPD       = '/path/to/PD';

parcelPattern = '*_parcelBOLD_z.mat'; % cortical atlas (here Schaefer2018_400Parcels_7networks) parcelwise BOLD timeseries
csfPattern    = '*_CSF_gBOLD_z.tsv';
nvoxPattern   = '*_parcel_nvox.tsv';   % optional (number of voxels per parcel)

out_root = '/path/to/parcelwise_coupling_outputs';
if ~exist(out_root,'dir'); mkdir(out_root); end

n_parc = 400;


%% Load in subjects

subjects = [];
subjects = [subjects; collect_group_parcel(rootControls, 'Control', parcelPattern, csfPattern, nvoxPattern)];
subjects = [subjects; collect_group_parcel(rootDLB,      'DLB',     parcelPattern, csfPattern, nvoxPattern)];
subjects = [subjects; collect_group_parcel(rootPD,       'PD',      parcelPattern, csfPattern, nvoxPattern)];

nSub = numel(subjects);
fprintf('Total subjects found with parcelBOLD+CSF: %d\n', nSub);
if nSub == 0; error('\nNo subjects found?...\n'); end

groupLabels = strings(nSub,1);
for s=1:nSub; groupLabels(s) = subjects(s).group; end
groupCats = categorical(groupLabels);
uGroups   = categories(groupCats);
nGroups   = numel(uGroups);

%% compute x-corr curves for each subject x parcel 
% allXC_parcel(s,p,:) = xcorr(parcelBOLD(:,p), csf, maxLagTR, 'coeff')

fprintf('\nComputing subject-level parcel xcorr curves...\n');

allXC_parcel = nan(nSub, n_parc, nLags, 'single');  
lags = -maxLagTR:maxLagTR;

for s = 1:nSub
    fprintf('  %3d/%d %s (%s)\n', s, nSub, subjects(s).id, subjects(s).group);

    % Load parcelBOLD
    S = load(subjects(s).parcel_file);
    if ~isfield(S,'parcelBOLD'); error('Missing variable parcelBOLD in %s', subjects(s).parcel_file); end
    parcelBOLD = double(S.parcelBOLD);

    % Load CSF
    csf = readmatrix(subjects(s).csf_file, 'FileType','text'); csf = csf(:);

    % Ensure correct orientation
    if size(parcelBOLD,2) ~= n_parc && size(parcelBOLD,1) == n_parc
        parcelBOLD = parcelBOLD.'; % T x 400
    end
    if size(parcelBOLD,2) ~= n_parc
        error('parcelBOLD not T x 400 in %s (got %d x %d)', subjects(s).parcel_file, size(parcelBOLD,1), size(parcelBOLD,2));
    end


    for p = 1:n_parc

        ts = parcelBOLD(:,p);

        if all(isnan(ts)) || std(ts,'omitnan') == 0
            continue;
        end
        if any(isnan(ts)); ts = fillmissing(ts,'linear','EndValues','nearest'); end

        xc = xcorr(ts, csf, maxLagTR, 'coeff');  % length nLags
        allXC_parcel(s,p,:) = single(xc(:));
    end
end

%% Get parcel-specific lags
fprintf('\nSelecting parcel-specific lags (grand mean of group means)...\n');

groupMeanXC_parcel = nan(nGroups, n_parc, nLags, 'single');

for g = 1:nGroups
    idxG = (groupCats == uGroups{g});
    groupMeanXC_parcel(g,:,:) = squeeze(mean(allXC_parcel(idxG,:,:), 1, 'omitnan')); % parcels x lags
end

grandMeanXC_parcel = squeeze(mean(groupMeanXC_parcel, 1, 'omitnan')); % parcels x lags

idxParcel    = nan(1,n_parc);
lagTRParcel  = nan(1,n_parc);
lagSecParcel = nan(1,n_parc);
minRParcel   = nan(1,n_parc);

for p = 1:n_parc
    curve = grandMeanXC_parcel(p,:);  % 1 x nLags
    if all(isnan(curve))
        continue;
    end
    [minR, idxMin] = min(curve);
    idxParcel(p)    = idxMin;
    lagTRParcel(p)  = lags(idxMin);
    lagSecParcel(p) = lags(idxMin) * TR;
    minRParcel(p)   = minR;
end


%% Compute coupling at parcel lag

fprintf('\nComputing coupling values at parcel-specific lags...\n');

couplingParcel = nan(nSub, n_parc);

for s = 1:nSub
    for p = 1:n_parc
        idx = idxParcel(p);
        if isnan(idx); continue; end
        xc = squeeze(allXC_parcel(s,p,:));
        if all(isnan(xc)); continue; end
        couplingParcel(s,p) = double(xc(idx));
    end
end

%% Get group means

groupMeanParcel = nan(nGroups, n_parc);
groupN          = zeros(nGroups,1);

for g = 1:nGroups
    idxG = (groupCats == uGroups{g});
    groupN(g) = sum(idxG);
    groupMeanParcel(g,:) = mean(couplingParcel(idxG,:), 1, 'omitnan');
end

grandMeanParcel = mean(couplingParcel, 1, 'omitnan');

%% Save outputs

outMat = fullfile(out_root, 'parcelwise_coupling_parcelSpecificLag_grandMeanOfGroupMeans.mat');
save(outMat, ...
    'couplingParcel','groupMeanParcel','grandMeanParcel', ...
    'idxParcel','lagTRParcel','lagSecParcel','minRParcel', ...
    'grandMeanXC_parcel','groupMeanXC_parcel','lags', ...
    'subjects','groupLabels','uGroups','groupN', ...
    'TR','maxLagSec','maxLagTR');

fprintf('\nSaved MAT: %s\n', outMat);

%% Write to CSV

grpTable = table(string(uGroups), groupN, 'VariableNames', {'Group','N'});
for p = 1:n_parc
    grpTable.(sprintf('P%03d', p)) = groupMeanParcel(:,p);
end
writetable(grpTable, fullfile(out_root, 'groupMeans_parcelCoupling_parcelSpecificLag.csv'));

%% Histogram of parcel lags (QC check)

figure('Name','Parcel-specific lags (s)');
histogram(lagSecParcel(~isnan(lagSecParcel)));
xlabel('Lag (s)'); ylabel('# parcels'); grid on;
title('Parcel-specific lag distribution');

%% all done!

fprintf('\n[done] Script finished.\n');

%% Helpers

function subjects = collect_group_parcel(rootDir, groupName, parcelPattern, csfPattern, nvoxPattern)
    subjects = [];
    subDirs = dir(fullfile(rootDir, 'sub-*'));
    subDirs = subDirs([subDirs.isdir]);

    fprintf('  [group %s] found %d subject dirs\n', groupName, numel(subDirs));

    for i = 1:numel(subDirs)
        subID  = subDirs(i).name;
        subPath = fullfile(rootDir, subID);

        parcelFiles = dir(fullfile(subPath, '**', parcelPattern));
        parcelFiles = parcelFiles(~startsWith({parcelFiles.name}, '._'));

        csfFiles = dir(fullfile(subPath, '**', csfPattern));
        csfFiles = csfFiles(~startsWith({csfFiles.name}, '._'));

        nvoxFiles = dir(fullfile(subPath, '**', nvoxPattern));
        nvoxFiles = nvoxFiles(~startsWith({nvoxFiles.name}, '._'));

        if isempty(parcelFiles) || isempty(csfFiles)
            fprintf('    [warn] %s: missing parcelBOLD or CSF file, skipping.\n', subID);
            continue;
        end

        s.id          = subID;
        s.group       = string(groupName);
        s.parcel_file = fullfile(parcelFiles(1).folder, parcelFiles(1).name);
        s.csf_file    = fullfile(csfFiles(1).folder, csfFiles(1).name);

        if ~isempty(nvoxFiles)
            s.nvox_file = fullfile(nvoxFiles(1).folder, nvoxFiles(1).name);
        else
            s.nvox_file = "";
        end

        subjects = [subjects; s]; 
    end
end
