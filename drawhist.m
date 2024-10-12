clc;clear;close all;
imglist = {'dim.jpeg','spot.jpeg'};
imglist = {'flatImage.png'};
for i = 1 : length(imglist)
    %img = imread(['../images/',char(imglist(i))]);
    img = imread([char(imglist(i))]);
    figure;
    subplot(2, 2, 1);
    imhist(rgb2gray(img));
    title('全色通道直方图',FontSize=14);
    
    subplot(2, 2, 2);
    imhist(img(:, :, 1));
    title('红色通道直方图',FontSize=14);
    
    subplot(2, 2, 3);
    imhist(img(:, :, 2));
    title('绿色通道直方图',FontSize=14);
    
    subplot(2, 2, 4);
    imhist(img(:, :, 3));
    title('蓝色通道直方图',FontSize=14);
end