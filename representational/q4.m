%% q4_combined.m
% 整合版：左边显示单个大图 (W_single)，右边显示2行5列10个小图 (W_top10)
clear; close all;

%% Load Data and Parameters
load('representational');  % 加载 Y 和 R
% 假设 R 为生成权重矩阵
[~, K_total] = size(W);

% 加载优化后的参数（假设文件中包含 opt_params, A, b）
load('result_q2/optimized_params.mat', 'opt_params', 'A', 'b');

%% 设置输出文件夹
output_folder = 'result_q4';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% Part 1: 获取单个分量的生成权重
single_k = 51;         % 选择一个分量
W_single = W(:, single_k);  % D x 1
D = size(W_single,1);
NumPix = sqrt(D);
if mod(NumPix,1)~=0
    error('像素数 D 必须为完全平方数');
end

%% Part 2: 选择生成权重较好的 10 个分量 (top10) 和 较差的 10 个分量 (bottom10)
% 这里以 A 的第 single_k 行作为衡量指标（仅为示例）
measure = A(single_k,:);         % 假设 A 尺寸为 [K_total x K_total]
[~, sorted_idx] = sort(measure, 'descend');
top10 = sorted_idx(1:10);        % 前 10 个分量索引
W_top10 = W(:, top10);           % 生成权重 [D x 10]

[~, sorted_idx_asc] = sort(measure, 'ascend');
bottom10 = sorted_idx_asc(1:10);  % 后 10 个分量索引
W_bottom10 = W(:, bottom10);     % 生成权重 [D x 10]

%% Part 3: 绘制整合图 - Top10
% 总网格为 2行7列：左侧大图占 2x2 = 4 个格子，右侧区域为 5列×2行 = 10 个格子
rows = 2; cols = 7;
% 设置外边距与间距（归一化坐标）
left = 0.01; right = 0.01; top = 0.08; bottom = 0.05;
hspace = 0.001; vspace = 0.07;

avail_width = 1 - left - right;
avail_height = 1 - top - bottom;
cell_width = (avail_width - (cols-1)*hspace) / cols;
cell_height = (avail_height - (rows-1)*vspace) / rows;

% 上排和下排 y 坐标
y_top = bottom + cell_height + vspace;
y_bottom = bottom;

% 设置图形窗口大小（高分辨率）
figure('Name','Generative Weights Combined - Top10','NumberTitle','off', ...
       'Position',[100, 100, 680, 200]);

%% 左侧大图：W_single 占据左侧 2列×2行（4个格子合并）
big_x = left; 
big_y = bottom;          
big_width = 2 * cell_width + hspace;    % 合并2列
big_height = 2 * cell_height + vspace;    % 合并2行

clim_single = max(abs(W_single))*[-1,1] + [-1e-5,1e-5];
ax_big = axes('Position',[big_x, big_y, big_width, big_height]);
imagesc(flipud(reshape(W_single, [NumPix, NumPix])), clim_single);
colormap(ax_big, gray);
axis(ax_big, 'off'); axis(ax_big, 'square');
title(ax_big, sprintf('k=%d', single_k), 'FontSize',12);

%% 右侧小图：W_top10 占据剩余区域（5列×2行 = 10个格子）
right_start_x = left + 2 * cell_width + hspace*2;
for r = 1:2
    for c = 1:5
        idx = (r-1)*5 + c;  % 当前小图索引（1~10）
        x_pos = right_start_x + (c-1)*(cell_width + hspace);
        if r == 1
            y_pos = y_top;
        else
            y_pos = y_bottom;
        end
        ax = axes('Position',[x_pos, y_pos, cell_width, cell_height]);
        Wcur = W_top10(:, idx);
        clim_cur = max(abs(Wcur))*[-1,1] + [-1e-5,1e-5];
        imagesc(flipud(reshape(Wcur, [NumPix, NumPix])), clim_cur);
        colormap(ax, gray);
        axis(ax, 'off'); axis(ax, 'square');
        title(ax, sprintf('j=%d', top10(idx)), 'FontSize',8);
    end
end

% 保存 Top10 图，高分辨率保存
top10_filename = fullfile(output_folder, sprintf('combined_generative_weights_top10_k=%d.png', single_k));
print(gcf, top10_filename, '-dpng', '-r300');
close(gcf);

%% Part 4: 绘制整合图 - Bottom10
figure('Name','Generative Weights Combined - Bottom10','NumberTitle','off', ...
       'Position',[100, 100, 680, 200]);

%% 左侧大图仍使用 W_single
ax_big = axes('Position',[big_x, big_y, big_width, big_height]);
imagesc(flipud(reshape(W_single, [NumPix, NumPix])), clim_single);
colormap(ax_big, gray);
axis(ax_big, 'off'); axis(ax_big, 'square');
title(ax_big, sprintf('k=%d', single_k), 'FontSize',12);

%% 右侧小图：绘制 W_bottom10，占据剩余 10 个格子
for r = 1:2
    for c = 1:5
        idx = (r-1)*5 + c;  % 当前小图索引（1~10）
        x_pos = right_start_x + (c-1)*(cell_width + hspace);
        if r == 1
            y_pos = y_top;
        else
            y_pos = y_bottom;
        end
        ax = axes('Position',[x_pos, y_pos, cell_width, cell_height]);
        Wcur = W_bottom10(:, idx);
        clim_cur = max(abs(Wcur))*[-1,1] + [-1e-5,1e-5];
        imagesc(flipud(reshape(Wcur, [NumPix, NumPix])), clim_cur);
        colormap(ax, gray);
        axis(ax, 'off'); axis(ax, 'square');
        title(ax, sprintf('j=%d', bottom10(idx)), 'FontSize',8);
    end
end

% 保存 Bottom10 图
bottom10_filename = fullfile(output_folder, sprintf('combined_generative_weights_bottom10_k=%d.png', single_k));
print(gcf, bottom10_filename, '-dpng', '-r300');
close(gcf);