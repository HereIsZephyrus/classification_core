clc;clear;close all;
%% init image and interface
global trueImage;
trueImage = imread("/Users/channingtong/Program/pattern_recorgnization/landuse/feature_city/truecolor.png");
global falseImage;
falseImage = imread("/Users/channingtong/Program/pattern_recorgnization/landuse/feature_city/falsecolor.png");
global rawImage;
rawImage = imread("/Users/channingtong/Program/pattern_recorgnization/landuse/feature_city/futureCity.tif");
global bandcompoent;
bandcompoent = true;
%figure;
global h;
h = imshow(trueImage);
mkdir('water');
mkdir('greenland');
mkdir('bareland');
mkdir('imprevious');
%% init UI
global xRange;
global yRange;
xRange = size(rawImage,2);
yRange = size(rawImage,1);
set(gcf,  'Position', [0 0 xRange yRange]);

global rectHandle;
rectHandle = [];
global counting;
counting.skip = 0;
counting.water = 0;
counting.greenland = 0;
counting.bareland = 0;
counting.imprevious = 0;
baseX = 70; baseY = 20; buttonWidth = 85; buttonHeight = 30; gapWidth = 100;
hChangeBand = uicontrol('Style', 'pushbutton', ...
                        'String', '更换波段', ...
                        'FontSize',14, ...
                        'Position',[baseX,baseY + 40,buttonWidth,buttonHeight],...
                        'Callback', @changeCallback);
hFinish = uicontrol('Style', 'pushbutton', ...
                        'String', '结束选择', ...
                        'FontSize',14, ...
                        'Position',[baseX,baseY,buttonWidth,buttonHeight],...
                        'Callback', @finishCallback);
global hSkip;
hSkip = uicontrol('Style', 'pushbutton', ...
                        'String', '跳过样本', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 1,baseY,buttonWidth,buttonHeight],...
                        'Callback', @skipCallback);
global hCountSkipLabel;
hCountSkipLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 1, baseY - 15, 100, 20], ...
                        'String', ['跳过数:',num2str(counting.skip)]);
global hWater;
hWater = uicontrol('Style', 'pushbutton', ...
                            'String', '这是水体', ...
                            'FontSize',14, ...
                            'Position',[baseX + gapWidth * 2,baseY,buttonWidth,buttonHeight],...
                            'Callback', @waterCallback);
global hCountWaterLabel;
hCountWaterLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 2, baseY - 15, 100, 20], ...
                        'String', ['水体数:',num2str(counting.water)]);
global hGreenland;
hGreenland = uicontrol('Style', 'pushbutton', ...
                        'String', '这是绿地', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 3,baseY,buttonWidth,buttonHeight],...
                        'Callback', @greenlandCallback);
global hCountGreenlandLabel;
hCountGreenlandLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 3, baseY - 15, 100, 20], ...
                        'String', ['绿地数:',num2str(counting.greenland)]);
global hBareland;
hBareland = uicontrol('Style', 'pushbutton', ...
                        'String', '这是裸地', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 4,baseY,buttonWidth,buttonHeight],...
                        'Callback', @barelandCallback);
global hCountBarelandLabel;
hCountBarelandLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 4, baseY - 15, 100, 20], ...
                        'String', ['裸地数:',num2str(counting.bareland)]);
global hImprevious;
hImprevious = uicontrol('Style', 'pushbutton', ...
                        'String', '这是建筑', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 5,baseY,buttonWidth,buttonHeight],...
                        'Callback', @impreviousCallback);
global hCountImpreviousLabel;
hCountImpreviousLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 5, baseY - 15, 100, 20], ...
                        'String', ['建筑数:',num2str(counting.imprevious)]);
global squareSize;
squareSize = 9;
global hSlider;
hSlider = uicontrol('Style', 'slider', ...
                        'Min', 5, ...
                        'Max', 40, ...
                        'Value', squareSize, ...
                        'Position', [baseX + gapWidth * 6, baseY + 5, 200, 20], ...
                        'Callback', @sliderCallback);
global hValueSizeLabel;
hValueSizeLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 6, baseY - 15, 100, 20], ...
                        'String', ['当前滑块尺寸:',num2str(squareSize)]);

global data;
data = table('Size',[0,5],'VariableTypes',{'double','double','double','double','categorical'},'VariableNames',{'xmin','ymin','width','height','label'});

%% main function
moveToNext();
%xpos = xRange - squareSize;
%ypos = yRange - squareSize;
%xStart = xpos - squareSize;
%xEnd = xpos + squareSize;
%yStart = ypos - squareSize;
%yEnd = ypos + squareSize;
%rectHandle = rectangle('Position', [xStart, yStart, xEnd-xStart, yEnd-yStart], ...
%                       'EdgeColor', 'r', 'LineWidth', 2);
%% functions
function finishCallback(~, ~)
    global data
    writetable(data,'classification.csv');
    close all
end
function waterCallback(~, ~)
    global xRange;global yRange;
    set(gca, 'XLim', [0,xRange], 'YLim', [0,yRange]);
    global rectHandle
    position = get(rectHandle,'Position');
    %position(1) = position(1) - 13;
    %position(2) = position(2) - 13;
    global data
    data = [data; table(position(1),position(2),position(3),position(4),categorical({'Water'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    global counting;
    counting.water = counting.water+1;
    for i = 1 : 4
        imwrite(imcrop(rawImage(:,:,i), position),['water/',num2str(counting.water),'.tif'],"WriteMode","append");
    end
    imwrite(imcrop(rawImage(:,:,1:3), position),['water/',num2str(counting.water),'.png']);
    global hCountWaterLabel;
    set(hCountWaterLabel, 'String', ['水体数:',num2str(counting.water)]);
    moveToNext();
end
function greenlandCallback(~, ~)
    global xRange;global yRange;
    set(gca, 'XLim', [0,xRange], 'YLim', [0,yRange]);
    global rectHandle
    position = get(rectHandle,'Position');
    %position(1) = position(1) - 13;
    %position(2) = position(2) - 13;
    global data
    data = [data; table(position(1),position(2),position(3),position(4),categorical({'Greenland'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    global counting;
    counting.greenland = counting.greenland+1;
    for i = 1 : 4
        imwrite(imcrop(rawImage(:,:,i), position),['greenland/',num2str(counting.greenland),'.tif'],"WriteMode","append");
    end
    imwrite(imcrop(rawImage(:,:,1:3), position),['greenland/',num2str(counting.greenland),'.png']);
    global hCountGreenlandLabel;
    set(hCountGreenlandLabel, 'String', ['绿地数:',num2str(counting.greenland)]);
    moveToNext();
end
function barelandCallback(~, ~)
    global xRange;global yRange;
    set(gca, 'XLim', [0,xRange], 'YLim', [0,yRange]);
    global rectHandle
    position = get(rectHandle,'Position');
    %position(1) = position(1) - 13;
    %position(2) = position(2) - 13;
    global data
    data = [data; table(position(1),position(2),position(3),position(4),categorical({'Bareland'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    global counting;
    counting.bareland = counting.bareland+1;
    for i = 1 : 4
        imwrite(imcrop(rawImage(:,:,i), position),['bareland/',num2str(counting.bareland),'.tif'],"WriteMode","append");
    end
    imwrite(imcrop(rawImage(:,:,1:3), position),['bareland/',num2str(counting.bareland),'.png']);
    global hCountBarelandLabel;
    set(hCountBarelandLabel, 'String', ['裸地数:',num2str(counting.bareland)]);
    moveToNext();
end
function impreviousCallback(~, ~)
    global xRange;global yRange;
    set(gca, 'XLim', [0,xRange], 'YLim', [0,yRange]);
    global rectHandle
    position = get(rectHandle,'Position');
    %position(1) = position(1) - 13;
    %position(2) = position(2) - 13;
    global data
    data = [data; table(position(1),position(2),position(3),position(4),categorical({'Imprevious'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    global counting;
    counting.imprevious = counting.imprevious+1;
    for i = 1 : 4
        imwrite(imcrop(rawImage(:,:,i), position),['imprevious/',num2str(counting.imprevious),'.tif'],"WriteMode","append");
    end
    imwrite(imcrop(rawImage(:,:,1:3), position),['imprevious/',num2str(counting.imprevious),'.png']);
    global hCountImpreviousLabel;
    set(hCountImpreviousLabel, 'String', ['建筑数:',num2str(counting.imprevious)]);
    moveToNext();
end
function skipCallback(~, ~)
    global counting;
    counting.skip = counting.skip + 1;
    global hCountSkipLabel;
    set(hCountSkipLabel, 'String', ['跳过数:',num2str(counting.skip)]);
    moveToNext();
end
function changeCallback(~, ~)
    global bandcompoent;
    global h;
    if (bandcompoent == true)
        global falseImage;
        set(h, 'CData', falseImage);
        bandcompoent = false;
    else
        global trueImage;
        set(h, 'CData', trueImage);
        bandcompoent = true;
    end
end

function sliderCallback(~, ~)
    global hSlider;
    global squareSize;
    global hValueSizeLabel;
    sliderValue = get(hSlider, 'Value');
    squareSize = round(sliderValue);
    set(hValueSizeLabel, 'String', ['当前值:',num2str(squareSize)]);
end
function moveToNext()
    global rectHandle;
    if ~isempty(rectHandle)
        delete(rectHandle);
    end
    global squareSize;
    global xpos;global ypos;
    global xRange;global yRange;
    xpos = randi([squareSize + 1, xRange - squareSize - 1]);
    ypos = randi([squareSize + 1, yRange - squareSize - 1]);
    xStart = xpos - squareSize;
    xEnd = xpos + squareSize;
    yStart = ypos - squareSize;
    yEnd = ypos + squareSize;
    rectHandle = rectangle('Position', [xStart, yStart, xEnd-xStart, yEnd-yStart], ...
                           'EdgeColor', 'r', 'LineWidth', 2);
    newXmin = max(1,xStart - squareSize * 10);
    newXLim = [newXmin, min(xRange,newXmin + squareSize * 21)];
    newYmin = max(1,yStart - squareSize * 10);
    newYLim = [newYmin, min(yRange,newYmin + squareSize * 21)];
    set(gca, 'XLim', newXLim, 'YLim', newYLim);
end