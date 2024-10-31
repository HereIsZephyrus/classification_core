clc;clear;close all;
figureUnits = 'centimeters';
figureWidth = 24;
figureHeight = 20;

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
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on
num = 8;
Max = zeros(1,num);
Std = zeros(1,num);
Mean = zeros(1,num);
for kind = 1 : num
    %% solve
    img = imread(['../results/',num2str(kind),'-temp.png']);
    [width,height,band] = size(img);
    widthCount = floor(width / bandwidth);
    heightCount = floor(height / bandwidth);
    
    bandDN = zeros(3,(width - bandwidth) * (height - bandwidth));
    
    weight = [0.299,0.587,0.114];
    for w = 1 : width - bandwidth
        for h = 1 : height - bandwidth
            startX = w;
            startY = h;
            ind = (w-1)*(height - bandwidth) + h;
            croppedImg = img(startX:startX+bandwidth-1,startY:startY+bandwidth-1,  :);
            for c = 1 : 3
                tempImg = double(reshape(croppedImg(:,:,c),1,bandwidth * bandwidth));
                tempImg(find(tempImg == 0)) = NaN;
                bandDN(c,ind) = mean(tempImg);
            end
        end
    end
    %% print
    scatter3(bandDN(1,:),bandDN(2,:),bandDN(3,:),2,colors(kind,:),'filled');
    plus = bandDN(1,:)+bandDN(2,:)+bandDN(3,:);
    Max(kind)= max(plus);
    Std(kind) = std(plus(~isnan(plus)));
    Mean(kind) = mean(plus(~isnan(plus)));
    axis([0 255 0 255 0 255]);
    %view([-10 25]);
    view([50 15]);
    disp(kind);
end
hold off
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
sgtitle('temp-total-RGBscattter',FontSize=18);
%print(figureHandle,'../statics/temp-total-scatter.png','-r300','-dpng');
%% statics
scatter3(Mean,Std,Max,100,'filled');
hold on 
scatterStr = cell(1,num);
for kind = 1 : 8
    scatterStr{kind} = num2str(kind);
end
%allRange = [0 500];
plane1x = [130 130 130 130];
fill3(plane1x, [0 500 500 0], [0 0 450 450],'r','FaceAlpha',0.1);
plane2y = [50 50 50 50];
fill3([0 500 500 0], plane2y, [0 0 450 450],'r','FaceAlpha',0.1);
plane3x = [240 240 240 240];
fill3(plane3x, [0 500 500 0], [0 0 450 450],'r','FaceAlpha',0.1);
plane4z = [450 450 450 450];
fill3([0 500 500 0], [0 0 500 500], plane4z,'b','FaceAlpha',0.1);
plane5x = [265 265 265 265];
fill3(plane5x, [0 500 500 0], [0 0 450 450],'r','FaceAlpha',0.1);
hold off
axis([0 max(Mean)+10 0 max(Std)+10 0 max(Max)+10]);
text(Mean,Std,Max + 15,scatterStr,FontSize=14);
xlabel("均值","FontSize",14);
ylabel("标准差","FontSize",14);
zlabel("最大值","FontSize",14);
sgtitle('temp-total-statics',FontSize=18);
%print(figureHandle,'../statics/temp-total-statics.png','-r300','-dpng');
hold off
%close all;
%hold off