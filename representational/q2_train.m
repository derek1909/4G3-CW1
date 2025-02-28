%% Script to minimize loss_and_grad and plot convergence

% Clear workspace and close figures
clear; close all; clc;

clear;
load('representational')
X = Y * R;
% X_subset = X;
X_subset = X(1:25600, :);
max_iter = 1000;  % maximum number of iterations (or line searches)

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
% Call minimize. The syntax is:
%   [opt_params, fX, iter] = minimize(initial_params, 'loss_and_grad', max_iter, X);
% where 'loss_and_grad' is the name of the function, and X is passed as an extra parameter.
[opt_params, fX, iter] = minimize(initial_params, 'loss_and_grad', max_iter, X_subset);

%--- Plot Convergence ---%
figure;
plot(fX, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Loss Function');
title('Convergence of the Loss Function');
grid on;

fprintf('Minimization completed in %d iterations.\n', iter);

%--- Reconstruct A and b from opt_params ---%
% opt_params is a (K^2 x 1) vector arranged as follows:
%   First K*(K-1) elements: log(a_{k,j}) for k ≠ j (row-major order)
%   Last K elements: log(b_k) for k = 1,...,K

numA = K*(K-1);
logA_vec = opt_params(1:numA);
logb     = opt_params(numA+1:end);

A = zeros(K, K);  % initialize A with zeros (diagonal remains zero)
idx = 1;
for k = 1:K
    for j = 1:K
        if k ~= j
            A(k,j) = exp(logA_vec(idx));
            idx = idx + 1;
        end
    end
end

b = exp(logb);  % b is a K x 1 vector

%--- Save optimized parameters along with A and b ---%
save('result_q2/optimized_params.mat', 'opt_params', 'A', 'b');

saveas(gcf, 'result_q2/convergence_plot.png');  % Save the current figure as a PNG file
close all;