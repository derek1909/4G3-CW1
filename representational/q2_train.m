%% Script to minimize loss_and_grad and plot convergence

% Clear workspace and close figures
clear; close all; clc;

clear;
load('representational')
X = Y * R;
% X_subset = X(:, :);
X_subset = X;

% If X is already defined, extract its dimensions:
[N, K] = size(X_subset);


%--- Initial Parameter Setup ---%
% For loss_and_grad, the parameters are:
%   - off-diagonals of a KxK matrix A (K*(K-1) parameters)
%   - a K-dimensional vector b (K parameters)
% Total number of parameters = K*(K-1) + K = K^2.
%
% Here we initialize all log-parameters to zero (i.e. A_offdiag = exp(0)=1 and b=exp(0)=1).
initial_params = zeros(K^2, 1);

%--- Minimization ---%
max_iter = 300;  % maximum number of iterations (or line searches)
% Call minimize. The syntax is:
%   [opt_params, fX, iter] = minimize(initial_params, 'loss_and_grad', max_iter, X);
% where 'loss_and_grad' is the name of the function, and X is passed as an extra parameter.
[opt_params, fX, iter] = minimize(initial_params, 'loss_and_grad', max_iter, X_subset);

%--- Plot Convergence ---%
figure;
plot(fX, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective Function Value');
title('Convergence of the Loss Function');
grid on;

fprintf('Minimization completed in %d iterations.\n', iter);

save('optimized_params.mat', 'opt_params');
