%% CALCULATE GLOBAL BOLD-CSF COUPLING

clear; clc;

%% Set parameters 

TR         = 3.0;      % seconds
maxLagSec  = 18;       % +/- seconds
maxLagTR   = round(maxLagSec / TR);

% Root folders for each group
rootControls = '/path/to/controls';
rootDLB      = '/path/to/DLB';
rootPD       = '/path/to/PD';

% Filename patterns for GM and CSF global z-scored signals
gmPattern  = '*GM_gBOLD_z.tsv';
csfPattern = '*CSF_gBOLD_z.tsv';

% Group labels 
groupNames = {'Control','DLB','PD'};

% Permutation settings
Nperm_global = 10000;   % for testing global coupling at chosen lag

%% Load in subjects

fprintf('Loading subjects...\n');

subjects = [];

% collect subjects from root folder
subjects = [subjects; collect_group(rootControls, 'Control', gmPattern, csfPattern)];
subjects = [subjects; collect_group(rootDLB,      'DLB',     gmPattern, csfPattern)];
subjects = [subjects; collect_group(rootPD,       'PD',      gmPattern, csfPattern)];

nSub = numel(subjects);
fprintf('Total subjects found with GM+CSF: %d\n', nSub);

if nSub == 0
    error('No subjects found. Check paths and patterns.');
end

%% Compute subject cross correlations

fprintf('\nComputing subject-level cross-correlations...\n');

nLags = 2*maxLagTR + 1;
allXC        = nan(nSub, nLags);   % gBOLD vs CSF
allXC_deriv  = nan(nSub, nLags);   % -d(gBOLD)/dt vs CSF (downward phases only)
peakR_sub    = nan(nSub, 1);
peakLagTR    = nan(nSub, 1);
groupLabels  = strings(nSub, 1);


for s = 1:nSub
    gm_file  = subjects(s).gm_file;
    csf_file = subjects(s).csf_file;
    grp      = subjects(s).group;
    groupLabels(s) = grp;

    % Read GM + CSF time series (column vectors)
    gm  = readmatrix(gm_file,  'FileType','text');
    csf = readmatrix(csf_file, 'FileType','text');
    gm  = gm(:);
    csf = csf(:);

    % Ensure equal length 
    T = min(numel(gm), numel(csf));
    if numel(gm) ~= numel(csf)
        warning('Subject %d (%s): GM (%d) and CSF (%d) differ, truncating to %d.', ...
            s, subjects(s).id, numel(gm), numel(csf), T);
    end
    gm  = gm(1:T);
    csf = csf(1:T);

    % Cross-correlation
    [xc_gm, lags] = xcorr(gm, csf, maxLagTR, 'coeff');
    allXC(s,:) = xc_gm(:).';

    % Subject-specific max negative peak for gBOLD–CSF
    [peakR, idxMin] = min(xc_gm);
    peakR_sub(s)    = peakR;
    peakLagTR(s)    = lags(idxMin);

    % negative derivative of gBOLD vs CSF 
    % First-order temporal derivative of gBOLD
    dgm = diff(gm);              % approx derivative: gm(t+1) - gm(t)

    % Take the -ve derivative and zero-out negative values:
    %   i.e. When gm is FALLING, dgm < 0  -> -dgm > 0, when gm is RISING, dgm > 0 -> -dgm < 0  (set to 0)
    neg_dgm = -dgm;
    neg_dgm(neg_dgm < 0) = 0;    % keep only downward phases

    % Match CSF length (T-1) to the derivative
    csf_trim = csf(1:end-1);

    % Cross-correlation between (−d gBOLD/dt)+ and CSF
    [xc_deriv, ~] = xcorr(neg_dgm, csf_trim, maxLagTR, 'coeff');
    allXC_deriv(s,:) = xc_deriv(:).';


    fprintf('Subject %2d/%d (%s): peak r = %.3f at lag %d TRs (%.2f s)\n', ...
        s, nSub, grp, peakR, lags(idxMin), lags(idxMin)*TR);
end

%% Get group-mean cross-correlation and compute grand mean (each group contributes equally)

fprintf('\nComputing group-mean cross-correlations and grand mean...\n');

groupCats = categorical(groupLabels);
uGroups   = categories(groupCats);   % Diagnosis groups
nGroups   = numel(uGroups);

groupMeanXC = nan(nGroups, nLags);

for g = 1:nGroups
    idxG = (groupCats == uGroups{g});
    groupMeanXC(g,:) = mean(allXC(idxG,:), 1, 'omitnan');
end

% Grand mean of group means 
grandMeanXC = mean(groupMeanXC, 1, 'omitnan');

% Select global lag as most negative point on the grand mean curve
[globalMinR, idxGlobal] = min(grandMeanXC);
globalLagTR  = lags(idxGlobal);
globalLagSec = globalLagTR * TR;

fprintf('\n Global Lag \n');
fprintf('Negative peak r = %.3f at lag = %d TRs (%.2f s)\n', ...
    globalMinR, globalLagTR, globalLagSec);

% Derivative-based cross-corr curves
groupMeanXC_deriv = nan(nGroups, nLags);
for g = 1:nGroups
    idxG = (groupCats == uGroups{g});
    groupMeanXC_deriv(g,:) = mean(allXC_deriv(idxG,:), 1, 'omitnan');
end

grandMeanXC_deriv = mean(groupMeanXC_deriv, 1, 'omitnan');


%% Get subject level coupling at negative lag

% Coupling per subject = xcorr(gm,csf) at -ve lag
couplingGlobal = allXC(:, idxGlobal);

% Print quick summary by group
fprintf('\nCoupling at global lag (%.2f s):\n', globalLagSec);
for g = 1:nGroups
    idxG = (groupCats == uGroups{g});
    m = mean(couplingGlobal(idxG), 'omitnan');
    med = median(couplingGlobal(idxG), 'omitnan');
    sd = std(couplingGlobal(idxG), 'omitnan');
    fprintf('  %-8s: mean = %.3f, median = %.3f, SD = %.3f, n = %d\n', ...
        uGroups{g}, m, med, sd, sum(idxG));
end

%% Global perm sig test

fprintf('\n[perm] Permutation test for global coupling at lag %.2f s...\n', globalLagSec);

obsMean = mean(couplingGlobal, 'omitnan');

% Null
nullMeans = nan(Nperm_global,1);
for p = 1:Nperm_global
    flips = (randi(2, nSub, 1) * 2 - 3);   % random +/-1
    nullMeans(p) = mean(couplingGlobal .* flips, 'omitnan');
end

% One-sided p ie. how often null mean <= observed mean (more negative)
p_global = (sum(nullMeans >= obsMean) + 1) / (Nperm_global + 1);

fprintf('Observed mean r = %.4f, p (one-sided, more negative) = %.4g (Nperm=%d)\n', ...
    obsMean, p_global, Nperm_global);



%% Plot curves

lagSec = lags * TR;

figure('Name','Grand mean cross-correlation with null band'); 
hold on;

% SEM
semXC = std(allXC, 0, 1) ./ sqrt(size(allXC,1));

% Grand mean with SEM 
errorbar(lagSec, grandMeanXC, semXC, ...
    'k.-', ...                 % black line + marker
    'LineWidth', 1.5, ...
    'MarkerSize', 25, ...
    'CapSize', 10);             % nicer error bars

% Global lag vertical line 
xline(globalLagSec, 'r--', 'LineWidth', 1.5); scatter(globalLagSec, globalMinR, 70, 'r', 'filled');

xlabel('Lag (s)');
ylabel('Cross-correlation (r)');
title(sprintf('gBOLD–CSF cross-correlation (global lag = %.2f s)', globalLagSec));

grid off;
box off;
hold off;


%%
semXC_deriv = std(allXC_deriv, 0, 1) ./ sqrt(size(allXC_deriv,1));


%% Plot -d (gBOLD/dt)+ – CSF cross-correlation 

figure('Name','Grand mean cross-correlation: -d(gBOLD)/dt vs CSF'); 
hold on;

% Error bars
errorbar(lagSec, grandMeanXC_deriv, semXC_deriv, ...
    'k.-', 'LineWidth', 1.5, 'MarkerSize', 25);


xlabel('Lag (s)', 'FontSize', 14);
ylabel('Cross-correlation (r)', 'FontSize', 14);


grid off; box off; hold off;


%% Quick boxplot of coupling at global lag per group 

figure('Name','Coupling at global lag per group');
boxplot(couplingGlobal, groupCats);
ylabel(sprintf('gBOLD–CSF coupling at lag = %.2f s', globalLagSec));
title('Group comparison of coupling strength');
grid on;

%% all done!

fprintf('\n Script finished.\n');


%% helper to find sub pattern

function subjects = collect_group(rootDir, groupName, gmPattern, csfPattern)


    subjects = [];
    rootDir = char(rootDir);  % ensure char for dir

    subDirs = dir(fullfile(rootDir, 'sub-*'));
    subDirs = subDirs([subDirs.isdir]);

    fprintf('  [group %s] found %d subject dirs\n', groupName, numel(subDirs));

    for i = 1:numel(subDirs)
        subID = subDirs(i).name;
        subPath = fullfile(rootDir, subID);

        % Find GM files
        gmFiles = dir(fullfile(subPath, '**', gmPattern));
        gmFiles = gmFiles(~startsWith({gmFiles.name}, '._'));

        % Find CSF files
        csfFiles = dir(fullfile(subPath, '**', csfPattern));
        csfFiles = csfFiles(~startsWith({csfFiles.name}, '._'));

        if isempty(gmFiles) || isempty(csfFiles)
            fprintf('    [warn] %s: missing GM or CSF file, skipping.\n', subID);
            continue;
        end

        if numel(gmFiles) > 1 || numel(csfFiles) > 1
            fprintf('    [warn] %s: multiple GM/CSF files found; using first match of each.\n', subID);
        end

        gm_file  = fullfile(gmFiles(1).folder,  gmFiles(1).name);
        csf_file = fullfile(csfFiles(1).folder, csfFiles(1).name);

        s.id       = subID;
        s.group    = string(groupName);
        s.gm_file  = gm_file;
        s.csf_file = csf_file;

        subjects = [subjects; s]; 
    end
end
