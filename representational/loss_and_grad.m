function [loss, grad] = loss_and_grad(params, X)
    % loss_and_grad computes the loss and its gradient with respect to the log-parameters.
    %
    % Inputs:
    %   params : vector of parameters [log(a_{k,j}); log(b_k)], where:
    %            - For the K×K matrix A, the diagonal is zero and the off-diagonals
    %              are given by a_{k,j} = exp(log(a_{k,j})) for k ~= j.
    %            - The vector b is given by b_k = exp(log(b_k)).
    %   X      : an N×K data matrix.
    %   K      : the number of outputs (columns of X and elements of b).
    %
    % Outputs:
    %   loss   : scalar loss value.
    %   grad   : vector of partial derivatives with respect to the input params,
    %            with the same size as params.
    
    [N, K] = size(X);
    
    % Determine number of parameters:
    numA = K*(K-1); % off-diagonals of A
    numB = K;        % entries of b
    
    % Extract parameters from the vector.
    logA_vec = params(1:numA);
    logb     = params(numA+1:end);
    
    % Reconstruct the K×K matrix A (with zeros on the diagonal).
    A = zeros(K, K);
    idx = 1;
    for k = 1:K
        for j = 1:K
            if k ~= j
                A(k,j) = exp(logA_vec(idx));
                idx = idx + 1;
            end
        end
    end
    % Compute vector b.
    b = exp(logb);
    
    % Compute X^{\odot2} (elementwise square).
    X2 = X.^2;  % size: N×K

    % Compute Sigma = X^{\odot2} * A^T + 1_N * b^T.
    Sigma = X2 * A' + b';  % size: N×K
    % Each entry: sigma_{n,k}^2.
    
    % Compute the loss.
    % Loss = (NK/2)*log(2*pi) + 1/2 * sum_{n,k} log(sigma_{n,k}^2) + 1/2 * sum_{n,k} (x_{n,k}^2 / sigma_{n,k}^2)
    loss = (N*K/2)*log(2*pi) + 0.5 * sum(log(Sigma(:))) + 0.5 * sum(sum(X2 ./ Sigma));
    
    % Compute helper matrix U.
    % For each element: U_{n,k} = 0.5*(1/sigma_{n,k}^2 - x_{n,k}^2/(sigma_{n,k}^4)).
    U = 0.5 * (  X2 ./ (Sigma.^2) - 1./Sigma);  % size: N×K
    
    % Compute gradients with respect to log(A) and log(b).
    % ∇_{log(A)} Loss = - ( (U^T * X^{\odot2}) ⊙ A )
    % ∇_{log(b)} Loss = - ( (U^T * 1_N) ⊙ b )
    grad_log_A_mat = - ( (U' * X2) .* A );      % size: K×K.
    grad_log_b     = - ( (U' * ones(N,1)) .* b );  % size: K×1.
    
    % Since A's diagonal is not parameterized, extract only off-diagonal gradients
    grad_log_A_vec = [];
    for k = 1:K
        for j = 1:K
            if k ~= j
                grad_log_A_vec = [grad_log_A_vec; grad_log_A_mat(k,j)];
            end
        end
    end
    
    % Stack gradients into one vector in the same order as params.
    grad = [grad_log_A_vec; grad_log_b];
    
    end