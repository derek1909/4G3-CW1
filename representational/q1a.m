%% Script to compute normalized variables, overlay Gaussian fits, and save excess kurtosis
clear; close all;

%--- Load Data and Optimized Parameters ---%
load('representational');  % should load variables Y and R
X = Y * R;
[N, K] = size(X);
NumPix = 32;

%--- Define Selected Indices ---%
selected_indices = [1, 3, 30, 227];

% If the list appears to be zero-indexed, convert to one-indexed.
if any(selected_indices == 0)
    warning('Detected zero-indexing. Converting to MATLAB one-indexing by adding 1.');
    selected_indices = selected_indices + 1;
end
selected_indices = selected_indices(selected_indices >= 1 & selected_indices <= K);

%--- Create Output Folder ---%
output_folder = 'result_q1a';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%--- Initialize storage for excess kurtosis results ---
ek_results = [];

%--- Plot, Overlay Gaussian, and Compute Excess Kurtosis for Each Selected Index ---%
for i = 1:length(selected_indices)
    k = selected_indices(i);
    data = X(:, k);
    
    % Compute excess kurtosis: sample kurtosis minus 3.
    ek = kurtosis(data) - 3;
    ek_results = [ek_results; k, ek];  % Store [component index, E.K.]
    

    %% Figure 1: Plot R(:,k) as a grayscale image
    figure('Units','inches','Position',[1, 1, 4, 4]);
    Rcur = R(:, k)';
    imagesc(reshape(Rcur', [NumPix, NumPix]), max(abs(Rcur))*[-1,1] + [-1e-5,1e-5]);
    set(gca, 'ylim', [1, NumPix], 'xlim', [1, NumPix], 'xtick', [], 'ytick', [], 'visible', 'off');
    axis square; colormap gray;
    filename = fullfile(output_folder, ['R_k=', num2str(k), '.png']);
    print(gcf, filename, '-dpng', '-r300');
    close(gcf);

    %% Figure 2: Plot W(:,k) as a grayscale image
    figure('Units','inches','Position',[1, 1, 4, 4]);
    Wcur = W(:, k)';
    imagesc(reshape(Wcur', [NumPix, NumPix]), max(abs(Wcur))*[-1,1] + [-1e-5,1e-5]);
    set(gca, 'ylim', [1, NumPix], 'xlim', [1, NumPix], 'xtick', [], 'ytick', [], 'visible', 'off');
    axis square; colormap gray;
    filename = fullfile(output_folder, ['W_k=', num2str(k), '.png']);
    print(gcf, filename, '-dpng', '-r300');
    close(gcf);

    %% Linear Scale Plot using histcounts (not histogram function)
    [counts, edges] = histcounts(data, 'Normalization', 'pdf');
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;
    
    figure('Units','inches','Position',[1, 1, 4, 4]);
    hold on;
    h1 = plot(binCenters, counts, 'o', 'MarkerSize', 4, ...
              'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
    hold off;
    xlabel(sprintf('$x_{%d}$', k), 'Interpreter', 'latex');
    ylabel(sprintf('$p(x_{%d})$', k), 'Interpreter', 'latex');
    legend([h1], {'Empirical p(x)'}, 'Location', 'best');
    axis square; grid on;
    
    filename = fullfile(output_folder, sprintf('normalized_marginal_linear_x%d.png', k));
    print(gcf, filename, '-dpng', '-r300');
    close(gcf);
    
    %% Log Scale Plot with Gaussian Overlay
    mu = mean(data);
    sigma_emp = std(data);
    x_vals = linspace(min(data), max(data), 1000);
    gauss_pdf = normpdf(x_vals, mu, sigma_emp);
    
    figure('Units','inches','Position',[1, 1, 4, 4]);
    hold on;
    h3 = plot(binCenters, counts, 'o', 'MarkerSize', 4, ...
              'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
    h4 = plot(x_vals, gauss_pdf, 'r-', 'LineWidth', 2);
    hold off;
    set(gca, 'YScale', 'log');
    xlabel(sprintf('$x_{%d}$', k), 'Interpreter', 'latex');
    ylabel(sprintf('$p(x_{%d})$', k), 'Interpreter', 'latex');
    legend([h3, h4], {'Empirical p(x)', 'Gaussian'}, 'Location', 'best');
    axis square; grid on;
    
    filename = fullfile(output_folder, sprintf('normalized_marginal_log_x%d.png', k));
    print(gcf, filename, '-dpng', '-r300');
    close(gcf);
end

%--- Save Excess Kurtosis Results in a Text File ---%
txt_filename = fullfile(output_folder, 'excess_kurtosis.txt');
fid = fopen(txt_filename, 'w');
fprintf(fid, 'Component\tExcess Kurtosis\n');
for j = 1:size(ek_results, 1)
    fprintf(fid, 'x_%d\t%f\n', ek_results(j, 1), ek_results(j, 2));
end
fclose(fid);

fprintf('All selected marginal distributions with Gaussian overlays have been saved.\n');
fprintf('Excess kurtosis results saved in %s\n', txt_filename);