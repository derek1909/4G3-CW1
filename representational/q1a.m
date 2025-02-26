clear
load('representational')
% Define the number of pixels per side (assuming each image is 32x32)
NumPix = 32;

% Choose the latent variable index k (for example, k = 227; change as needed)
k = 227; % 1, 3, 30, 227

% Compute the latent variable activations for the chosen k
xk = Y * R(:, k);

% Compute Gaussian parameters for xk
mu = mean(xk);
sigma = std(xk);
x_vals = linspace(min(xk), max(xk), 100);
gauss_pdf = normpdf(x_vals, mu, sigma);

% Compute histogram data as points (normalized to PDF)
[counts, edges] = histcounts(xk, 'Normalization', 'pdf');
binCenters = (edges(1:end-1) + edges(2:end))/2;

%% Figure 1: Plot R(:,k) as a grayscale image
figure('Units','inches','Position',[1, 1, 4, 4]);
Rcur = R(:, k)';
imagesc(reshape(Rcur', [NumPix, NumPix]), max(abs(Rcur))*[-1,1] + [-1e-5,1e-5]);
set(gca, 'ylim', [1, NumPix], 'xlim', [1, NumPix], 'xtick', [], 'ytick', [], 'visible', 'off');
axis square; colormap gray;
print(gcf, ['R_k=', num2str(k), '.png'], '-dpng', '-r300');

%% Figure 2: Plot W(:,k) as a grayscale image
figure('Units','inches','Position',[1, 1, 4, 4]);
Wcur = W(:, k)';
imagesc(reshape(Wcur', [NumPix, NumPix]), max(abs(Wcur))*[-1,1] + [-1e-5,1e-5]);
set(gca, 'ylim', [1, NumPix], 'xlim', [1, NumPix], 'xtick', [], 'ytick', [], 'visible', 'off');
axis square; colormap gray;
print(gcf, ['W_k=', num2str(k), '.png'], '-dpng', '-r300');

%% Figure 3: Plot histogram (linear scale) as points with Gaussian overlay
figure('Units','inches','Position',[1, 1, 4, 4]);
hold on;
h1 = plot(binCenters, counts, 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
% h2 = plot(x_vals, gauss_pdf, 'r-', 'LineWidth', 2);
hold off;
xlabel('$x_k$', 'Interpreter', 'latex');
ylabel('$p(x_k)$', 'Interpreter', 'latex');
legend([h1], {'Empirical p(x)'}, 'Location', 'best');
axis square;
print(gcf, ['hist_linear_k=', num2str(k), '.png'], '-dpng', '-r300');

%% Figure 4: Plot histogram (log scale) as points with Gaussian overlay
figure('Units','inches','Position',[1, 1, 4, 4]);
hold on;
h3 = plot(binCenters, counts, 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
h4 = plot(x_vals, gauss_pdf, 'r-', 'LineWidth', 2);
hold off;
set(gca, 'YScale', 'log');
xlabel('$x_k$', 'Interpreter', 'latex');
ylabel('$p(x_k)$', 'Interpreter', 'latex');
legend([h3, h4], {'Empirical p(x)', 'Gaussian'}, 'Location', 'best');
axis square;
print(gcf, ['hist_log_k=', num2str(k), '.png'], '-dpng', '-r300');
close all;
