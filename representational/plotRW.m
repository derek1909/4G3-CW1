function plotIm(W)
  % function plotIm(W)
  %
  % Plots square images by reshaping the columns of W into images.
  %
  % INPUTS:
  %   W = generative weights [D,K]

  [D,K] = size(W);  
  NumPlots = ceil(sqrt(K));
  NumPix = sqrt(D);
  
  top = 0.05;
  bottom = 0.05;
  left = 0.05;
  right = 0.05;
  vspace = 0.02;
  hspace = 0.02;
  
  width = (1-left-right-hspace*(NumPlots-1))/NumPlots;
  height = (1-top-bottom-vspace*(NumPlots-1))/NumPlots;
  
  across = [width+hspace,0,0,0]';
  down = -[0,height+vspace,0,0]';
  
  pos = zeros(4,NumPlots,NumPlots);
  
  for d1=1:NumPlots
    for d2=1:NumPlots
      pos(:,d1,d2) = [left, 1-top-height,width,height]' + (d1-1)*across+(d2-1)*down;
    end
  end
  
  pos = reshape(pos,[4,NumPlots*NumPlots]);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Fix size of figure so that the patches are square
  ScrSz = get(0, 'ScreenSize');
  hFrac = 0.8;
  hFig = ScrSz(4)*hFrac;
  wFig = height/width*hFig;
  posFig = [ScrSz(3)/2-wFig/2, ScrSz(4)/2-hFig/2, wFig, hFig];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  figure1 = figure('Name','Weights','NumberTitle','off','position',posFig);
  
  for k=1:K
    axk = axes('position',pos(:,k));
    hold on;
    
    Wcur = W(:,k)';
    imagesc(reshape(Wcur',[NumPix,NumPix]), max(abs(Wcur))*[-1,1] + [-1e-5,1e-5])
    set(gca, 'ylim', [1,NumPix], 'xlim', [1,NumPix], 'xtick', [], 'ytick', []);
    axis square;
    colormap gray
    % Adjust the text placement: move further below the image and disable clipping.
    text(0.5, -0.1, num2str(k), 'Units', 'normalized', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'Color', 'black', 'FontSize', 8, 'Clipping', 'off');
    hold off;
  end  
end