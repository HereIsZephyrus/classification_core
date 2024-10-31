clc;clear;close all;
figureUnits = 'centimeters';
figureWidth = 48;
figureHeight = 26;

imgdim = imread('../images/dim.jpeg');
imgspot = imread('../images/spot.jpeg');
flash = {'spot','dim'};
fruit = {'yellowpeach','grape','blackplum','apple','peach','dongzao'};
colors = [
    1.0000    0.8510    0.1843;
    0.7490         0    0.7490;
    0    0.5451    0.5451;
    1       0       0;
    1.0000    0.7529    0.7961;
    0.5490    0.4275    0.1922;
    0.4667    0.5333    0.6000;
    0.1176    0.5647    1.0000;
    0.9412    0.0078    0.4980;
    ];
convcolor = [ 0.9608    1.0000    0.9804];
bandwidth = 5;
colormap = addcolorplus(316);
%for i = 1 : length(flash)
%type = flash{i};
%for kind = 1 : length(fruit)
for kind = 1 : 8
    %% solve
    %imgName = char(strcat(fruit(kind),'-',type));
    %img = imread(['../images/',imgName,'.png']);
    img = imread(['../results/',num2str(kind),'.png']);
    [width,height,band] = size(img);
    widthCount = floor(width / bandwidth);
    heightCount = floor(height / bandwidth);
    
    bandDN = zeros(3,widthCount * heightCount);
    
    weight = [0.299,0.587,0.114];
    for w = 1 : width - bandwidth
        for h = 1 : height - bandwidth
            startX = w;
            startY = h;
            ind = (w-1)*heightCount + h;
            croppedImg = img(startX:startX+bandwidth-1,startY:startY+bandwidth-1,  :);
            for c = 1 : 3
                tempImg = double(reshape(croppedImg(:,:,c),1,bandwidth * bandwidth));
                tempImg(find(tempImg == 0)) = NaN;
                bandDN(c,ind) = mean(tempImg);
            end
        end
    end
    %rband = bandDN(1,:);
    %rband = rband(~isnan(rband));
    %gband = bandDN(1,:);
    %gband = gband(~isnan(gband));
    %bband = bandDN(1,:);
    %bband = bband(~isnan(bband));
    %K = convhull(rband, gband, bband);
    %% print
    figureHandle = figure;
    set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
    tiledlayout(2,3,"TileSpacing","tight",'Padding','compact');
    nexttile(1);
    scatter(bandDN(1,:),bandDN(2,:),2,colors(kind,:),'filled');
    xlabel("红色通道","FontSize",14);
    ylabel("绿色通道","FontSize",14);
    axis([0 255 0 255]);
    xticks(0:51:255);
    yticks(0:51:255);
    nexttile(2);
    scatter(bandDN(1,:),bandDN(3,:),2,colors(kind,:),'filled');
    xlabel("红色通道","FontSize",14);
    ylabel("蓝色通道","FontSize",14);
    axis([0 255 0 255]);
    xticks(0:51:255);
    yticks(0:51:255);
    nexttile(3);
    scatter(bandDN(2,:),bandDN(3,:),2,colors(kind,:),'filled');
    xlabel("红色通道","FontSize",14);
    ylabel("蓝色通道","FontSize",14);
    axis([0 255 0 255]);
    xticks(0:51:255);
    yticks(0:51:255);
    nexttile(4);
    scatter3(bandDN(1,:),bandDN(2,:),bandDN(3,:),2,colors(kind,:),'filled');
    axis([0 255 0 255 0 255]);
    view([-10 25]);
    xticks(0:51:255);
    yticks(0:51:255);
    zticks(0:51:255);
    xlabel('红色通道',FontSize=14);
    ylabel('绿色通道',FontSize=14);
    zlabel('蓝色通道',FontSize=14);
    nexttile(5);
    scatter3(bandDN(1,:),bandDN(2,:),bandDN(3,:),2,colors(kind,:),'filled');
    axis([0 255 0 255 0 255]);
    view([45 15]);
    xticks(0:51:255);
    yticks(0:51:255);
    zticks(0:51:255);
    xlabel('红色通道',FontSize=14);
    ylabel('绿色通道',FontSize=14);
    zlabel('蓝色通道',FontSize=14);
    nexttile(6);
    scatter3(bandDN(1,:),bandDN(2,:),bandDN(3,:),2,colors(kind,:),'filled');
    axis([0 255 0 255 0 255]);
    view([160 -10]);
    xticks(0:51:255);
    yticks(0:51:255);
    zticks(0:51:255);
    xlabel('红色通道',FontSize=14);
    ylabel('绿色通道',FontSize=14);
    zlabel('蓝色通道',FontSize=14);
    %sgtitle([imgName,'-RGBscattter'],FontSize=18);
    %sgtitle(['temp-',num2str(kind),'-RGBscattter'],FontSize=18);
    figW = figureWidth;
    figH = figureHeight;
    set(figureHandle,'PaperUnits',figureUnits);
    set(figureHandle,'PaperPosition',[0 0 figW figH]);
    %print(figureHandle,['../statics/',imgName,'-scatter.png'],'-r300','-dpng');
    %print(figureHandle,['../statics/temp-',num2str(kind),'-scatter.png'],'-r300','-dpng');
    disp(kind);
end
%end
close all;
%hold off