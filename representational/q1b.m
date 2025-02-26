% Choose indices for the pair of components (example values)
X = Y * R;
k1 = 50;  % Example latent component index for k1
k2 = 51; % Example latent component index for k2

% Extract the latent variable values for k1 and k2 from the latent representation matrix X (size N x K)
xk1 = X(:, k1);  
xk2 = X(:, k2);

% Define the number of bins for the 2D histogram
numBins = 100;
edges1 = linspace(min(xk1), max(xk1), numBins+1);
edges2 = linspace(min(xk2), max(xk2), numBins+1);

% Compute 2D histogram counts (joint distribution estimate)
[counts,~,~,~] = histcounts2(xk1, xk2, edges1, edges2, 'Normalization', 'count');

% Renormalize each slice (each bin along xk1) to obtain the conditional distribution p(xk2|xk1)
conditional = zeros(size(counts));
for i = 1:size(counts,1)
    rowSum = sum(counts(i,:));
    if rowSum > 0
        conditional(i,:) = counts(i,:) / rowSum;
    end
end

% Plot the conditional distribution using imagesc
figure;
imagesc(edges1, edges2, conditional');
set(gca, 'YDir', 'normal');  % Ensures y-axis increases upward
xlabel('$x_{k_1}$','Interpreter','latex');
ylabel('$x_{k_2}$','Interpreter','latex');
colorbar;
title('Conditional Distribution $p(x_{k_2}|x_{k_1})$','Interpreter','latex');


% Define the folder name
folderName = 'result_q1b';

% Create the folder if it does not exist
if ~exist(folderName, 'dir')
    mkdir(folderName);
end

% Define the save path and file name
savePath = fullfile(folderName, ['conditional_k1=',num2str(k1),'_k2=',num2str(k2),'.png']);

% Save the figure as a high-resolution PNG file
print(gcf, savePath, '-dpng', '-r300');



close all;