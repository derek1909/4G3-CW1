%% Script to compute and compare excess kurtosis for X and C
clear; close all;

%% Load Data and Optimized Parameters
% Load representational data (should load Y and R)
load('representational');  % loads Y and R
X = Y * R;               % latent representation matrix, size: N x K
[N, K] = size(X);

% Load optimized parameters from the result_q2 folder.
% The MAT-file is expected to contain: opt_params, A, and b.
load('result_q2/optimized_params.mat', 'opt_params', 'A', 'b');

%% Compute Normalized Variables C
% Compute elementwise square of X.
X2 = X.^2;
% Compute Sigma = X^{\odot2} * A' + ones(N,1)*b'
Sigma = X2 * A' + repmat(b', N, 1);  % Sigma is N x K
% Compute sample-wise standard deviations.
sigma = sqrt(Sigma);
% Compute normalized variables: c_{n,k} = x_{n,k} / sigma_{n,k}
C = X ./ sigma;

%% Calculate Excess Kurtosis for Each Component
% For X, compute kurtosis for each column and subtract 3.
EK_X = kurtosis(X) - 3;  % 1 x K vector
% For C, compute kurtosis for each column and subtract 3.
EK_C = kurtosis(C) - 3;  % 1 x K vector

%% Plot the Excess Kurtosis Curves
figure;
plot(1:K, EK_X, 'b-', 'LineWidth', 2, 'MarkerSize', 6); hold on;
plot(1:K, EK_C, 'r-', 'LineWidth', 2, 'MarkerSize', 6); hold off;
set(gca, 'YScale', 'log');

xlabel('Component Index, $k$', 'Interpreter', 'latex');
ylabel('Excess Kurtosis', 'Interpreter', 'latex');
legend({'$X$', '$C$'}, 'Interpreter', 'latex', 'Location', 'best');
title('Excess Kurtosis for $X$ and Normalized $C$', 'Interpreter', 'latex');
grid on;
axis tight;

%% Save the Figure
output_folder = 'result_q3';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
saveas(gcf, fullfile(output_folder, 'excess_kurtosis_comparison.png'));
fprintf('Excess kurtosis comparison plot saved to %s\n', fullfile(output_folder, 'excess_kurtosis_comparison.png'));