clc;clear;close all;
imgdim = imread('../images/dim.jpeg');
imgspot = imread('../images/spot.jpeg');
bandwidth = 25;
widthCount = floor(width / bandwidth);
heightCount = floor(height / bandwidth);

bandDN = zeros(3,widthCount * heightCount);
light = zeros(1,widthCount * heightCount);

img = imgspot;
weight = [0.299,0.587,0.114];
for w = 1 : widthCount-1
    for h = 1 : heightCount-1
        startX = (w-1) * bandwidth + 1;
        startY = (h-1) * bandwidth + 1;
        ind = (w-1)*heightCount + h;
        croppedImg = img(startX:startX+bandwidth-1,startY:startY+bandwidth-1,  :);
        bandDN(1,ind) = mean(reshape(croppedImg(:,:,1),1,bandwidth * bandwidth));
        bandDN(2,ind) = mean(reshape(croppedImg(:,:,2),1,bandwidth * bandwidth));
        bandDN(3,ind) = mean(reshape(croppedImg(:,:,3),1,bandwidth * bandwidth));
        light(ind) = weight * [bandDN(:,ind)];
    end
end
colormap = addcolorplus(316);
%%
scatter3(bandDN(1,:),bandDN(2,:),bandDN(3,:),1,light,'filled');
hold on
vertices = [
    0 0 0;
    255 255 0;
    255 255 255;
    255 0 0;
    ];
faces = [
    1 2 3;
    2 3 4;
    1 2 4;
    1 3 4;
    ];
%plot3([0 255],[0 255],[0 255],'r-','LineWidth',2);
%h = patch('Vertices', vertices, 'Faces', faces, ...
%           'FaceColor', 'r', ...         
%           'FaceAlpha', 0.1, ...         
%           'EdgeColor', 'r');            
hold off
axis([0 255 0 255 0 255]);
xticks(0:51:255);
yticks(0:51:255);
zticks(0:51:255);
c = colorbar;
title(c, '亮度'); 
annotation("textbox",[0.2 0.3 0.2 0.4], ...
    'String',['bandwidth=',num2str(bandwidth)], ...
    'BackgroundColor','w','EdgeColor','b', ...
    'FontSize',14,'FitBoxToText','on');
xlabel('红色通道',FontSize=14);
ylabel('绿色通道',FontSize=14);
zlabel('蓝色通道',FontSize=14);