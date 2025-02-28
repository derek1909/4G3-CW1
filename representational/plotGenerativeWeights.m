
function plotGenerativeWeights(W_plot, figTitle)
    % plotGenerativeWeights - 绘制矩阵 W_plot 的各列为一幅图像，并以网格形式排列。
    %
    % 语法:
    %   plotGenerativeWeights(W_plot, figTitle)
    %
    % 输入:
    %   W_plot   - 大小为 [D x K_plot] 的矩阵，每一列代表一幅图像。D 必须为一个完全平方数，
    %              即图像尺寸为 sqrt(D) x sqrt(D)。
    %   figTitle - 图形窗口的标题。
    %
    % 示例:
    %   plotGenerativeWeights(W, 'Generative Weights');
    %
    % 注意:
    %   如果 D 不是完全平方数，则函数会报错。
    
    % 检查像素数是否为完全平方数
    D = size(W_plot, 1);
    K_plot = size(W_plot, 2);
    NumPix = sqrt(D);
    if mod(NumPix, 1) ~= 0
        error('像素个数必须为完全平方数。');
    end
    
    % 计算网格行列数，使得可以容纳所有图像
    NumPlots = ceil(sqrt(K_plot));
    
    % 设置边距和间隔
    top = 0.05; bottom = 0.05; left = 0.05; right = 0.05; 
    vspace = 0.01; hspace = 0.01;
    
    width = (1 - left - right - hspace*(NumPlots-1)) / NumPlots;
    height = (1 - top - bottom - vspace*(NumPlots-1)) / NumPlots;
    across = [width+hspace, 0, 0, 0]';
    down = -[0, height+vspace, 0, 0]';
    
    % 构造每个子图的绘图位置（以归一化单位表示）
    pos = zeros(4, NumPlots, NumPlots);
    for d1 = 1:NumPlots
        for d2 = 1:NumPlots
            pos(:, d1, d2) = [left; 1 - top - height; width; height] + (d1-1)*across + (d2-1)*down;
        end
    end
    pos = reshape(pos, [4, NumPlots*NumPlots]);
    
    % 调整图形窗口大小，使得每个图像块为正方形
    ScrSz = get(0, 'ScreenSize'); 
    hFrac = 0.8; 
    hFig = ScrSz(4) * hFrac; 
    wFig = height/width * hFig;
    posFig = [ScrSz(3)/2 - wFig/2, ScrSz(4)/2 - hFig/2, wFig, hFig];
    
    figure('Name', figTitle, 'NumberTitle', 'off', 'Position', posFig);
    
    % 对每个分量绘图
    for k = 1:K_plot
        axk = axes('Position', pos(:, k));
        hold(axk, 'on');
        Wcur = W_plot(:, k);
        % 根据当前图像的最大绝对值设定颜色范围
        clim = max(abs(Wcur)) * [-1, 1] + [-1e-5, 1e-5];
        imagesc(reshape(Wcur, [NumPix, NumPix]), clim);
        set(axk, 'YLim', [1, NumPix], 'XLim', [1, NumPix], ...
                 'XTickLabel', '', 'YTickLabel', '', 'Visible', 'off');
        colormap(axk, gray);
        hold(axk, 'off');
    end
    
    drawnow;