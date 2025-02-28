%% Script to compare conditional distributions p(c_{k2}|c_{k1}) and p(x_{k2}|x_{k1})
clear; close all;

%% Load Data and Optimized Parameters
load('representational');  % should load variables Y and R
X = Y * R;               % latent representation (N x K)

% Use all samples or a subset if desired:
X_subset = X;  % or e.g., X(1:10000, :)

% Load the optimized parameters (assumed to contain opt_params, A, and b)
load('result_q2/optimized_params.mat', 'opt_params', 'A', 'b');

[N, K] = size(X_subset);

%% Compute Variance and Normalized Variables
% Compute elementwise square of X
X2 = X_subset.^2;

% Compute Sigma = X^{\odot2} * A' + ones(N,1)*b'
Sigma = X2 * A' + repmat(b', N, 1);   % (N x K)
sigma = sqrt(Sigma);                  % standard deviations

% Compute normalized latent variables: C = X ./ sigma
C = X_subset ./ sigma;

%% Set Indices for Comparison
% Choose indices for the pair of components (example values)
k1 = 1;  % latent component index for k1
k2 = 51;  % latent component index for k2

% Extract latent variable values for x and normalized values for c
xk1 = X_subset(:, k1);
xk2 = X_subset(:, k2);

ck1 = C(:, k1);
ck2 = C(:, k2);

%% Plot Conditional Distribution for x: p(x_{k2}| x_{k1})
numBins = 100;  % number of bins for the 2D histogram

% Define bin edges for xk1 and xk2
edges1_x = linspace(min(xk1), max(xk1), numBins+1);
edges2_x = linspace(min(xk2), max(xk2), numBins+1);

% Compute 2D histogram counts for (xk1, xk2)
[counts_x, ~, ~, ~] = histcounts2(xk1, xk2, edges1_x, edges2_x, 'Normalization', 'count');

% Renormalize each row to obtain the conditional distribution p(x_{k2}| x_{k1})
conditional_x = zeros(size(counts_x));
for i = 1:size(counts_x,1)
    rowSum = sum(counts_x(i,:));
    if rowSum > 0
        conditional_x(i,:) = counts_x(i,:) / rowSum;
    end
end

% Plot p(x_{k2}| x_{k1})
figure;
imagesc(edges1_x, edges2_x, conditional_x');
set(gca, 'YDir', 'normal');  % Ensure y-axis increases upward
xlabel('$x_{k_1}$','Interpreter','latex');
ylabel('$x_{k_2}$','Interpreter','latex');
colorbar;
title('Conditional Distribution $p(x_{k_2}|x_{k_1})$','Interpreter','latex');

% Create folder if it does not exist
folderName = 'result_q3b';
if ~exist(folderName, 'dir')
    mkdir(folderName);
end
% Save the figure as a high-resolution PNG file
savePath_x = fullfile(folderName, sprintf('conditional_x_k1=%d_k2=%d.png', k1, k2));
print(gcf, savePath_x, '-dpng', '-r300');
close(gcf);

%% Plot Conditional Distribution for c: p(c_{k2}| c_{k1})
% Define bin edges for ck1 and ck2
edges1_c = linspace(min(ck1), max(ck1), numBins+1);
edges2_c = linspace(min(ck2), max(ck2), numBins+1);

% Compute 2D histogram counts for (ck1, ck2)
[counts_c, ~, ~, ~] = histcounts2(ck1, ck2, edges1_c, edges2_c, 'Normalization', 'count');

% Renormalize each row to obtain the conditional distribution p(c_{k2}| c_{k1})
conditional_c = zeros(size(counts_c));
for i = 1:size(counts_c,1)
    rowSum = sum(counts_c(i,:));
    if rowSum > 0
        conditional_c(i,:) = counts_c(i,:) / rowSum;
    end
end

% Plot p(c_{k2}| c_{k1})
figure;
imagesc(edges1_c, edges2_c, conditional_c');
set(gca, 'YDir', 'normal');  % Ensure y-axis increases upward
xlabel('$c_{k_1}$','Interpreter','latex');
ylabel('$c_{k_2}$','Interpreter','latex');
colorbar;
title('Conditional Distribution $p(c_{k_2}|c_{k_1})$','Interpreter','latex');

% Save the normalized conditional plot
savePath_c = fullfile(folderName, sprintf('conditional_c_k1=%d_k2=%d.png', k1, k2));
print(gcf, savePath_c, '-dpng', '-r300');
close(gcf);

%% Observations
% At this point, you can compare the two saved figures:
%   - conditional_x_k1=50_k2=51.png (for p(x_{k2}| x_{k1}))
%   - conditional_c_k1=50_k2=51.png (for p(c_{k2}| c_{k1}))
%
% Typically, you may notice that the conditional distribution for the normalized
% variables p(c_{k2}| c_{k_1}) is more standardized (e.g., more symmetric or with 
% reduced heteroscedasticity) compared to p(x_{k2}| x_{k_1}). This indicates that the 
% normalization procedure (dividing by the local standard deviation) helps remove scale 
% differences and stabilizes the conditional variance across different values of the 
% conditioning variable.
%
% Print your observations based on the visual comparison of these figures.
fprintf('Conditional distributions for x and c have been saved to:\n%s\n%s\n', savePath_x, savePath_c);