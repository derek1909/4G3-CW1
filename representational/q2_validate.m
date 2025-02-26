%% Example Script to Check Gradients for loss_and_grad
clear;
load('representational')
X = Y * R;
X_subset = X(1:10, 1:50);

% If X is already defined, extract its dimensions:
[N, K] = size(X_subset);

% Set a small perturbation for finite differences:
epsilon = 1e-6;

% Number of parameters:
% For the KxK matrix A (with zeros on the diagonal) there are K*(K-1) parameters,
% and for the vector b there are K parameters.
numA = K * (K - 1);
numB = K;

% Initialize the parameter vector (stacking [log(a_{k,j}); log(b_k)]), here set to zero.
% This corresponds to A having off-diagonals exp(0)=1 and b=exp(0)=1.
init_params = zeros(numA + numB, 1);

a = size(X_subset)

% Call checkgrad on the function 'loss_and_grad'.
% The additional parameters passed to loss_and_grad are the data matrix X and the number K.
d = checkgrad('loss_and_grad', init_params, epsilon, X_subset );

% Print the relative error computed by checkgrad.
fprintf('Relative error from checkgrad: %e\n', d);