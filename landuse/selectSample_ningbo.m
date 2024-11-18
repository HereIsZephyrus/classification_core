clc;clear;close all;
global bandcompoent;
global trueImageBuddle;
global falseImageBuddle;
global bandSeries
global yearSeries
bandcompoent = true;
itcps = [0.3, 8.8, 6.1, 41.2, 25.4, 17.2];
slopes = [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071];
%% init image and interface
dataFolderName = "/Users/channingtong/Program/pattern_recorgnization/landuse/ningbo/data";
yearSeries = 1997:5:2022;
bandSeries = cell(6,6);
QASeries = cell(6,1);
TMbandname = {'1','2','3','4','5','7'};
OLIbandname = {'2','3','4','5','6','7'};
for i = 1:6
    year = yearSeries(i);
    yearName = ['/Users/channingtong/Program/pattern_recorgnization/landuse/ningbo/data/' , int2str(year),'/'];
    QAPath = dir([yearName,'*QA.TIF']);
    QAname = [QAPath.folder,'/',QAPath.name];
    QASeries{i} = imread(QAname);
    if (year < 2015)
        for j = 1 : 6
            fileSuffix = ['B',TMbandname{j},'.TIF'];
            imagePath = dir([yearName,'*',fileSuffix]);
            imagePath = [imagePath.folder,'/',imagePath.name];
            bandSeries{i,j} = imread(imagePath);
            bandSeries{i,j} = bandSeries{i,j} * slopes(j) + itcps(i);
        end
    else
        for j = 1 : 6
            fileSuffix = ['B',OLIbandname{j},'.TIF'];
            imagePath = dir([yearName,'*',fileSuffix]);
            imagePath = [imagePath.folder,'/',imagePath.name];
            bandSeries{i,j} = imread(imagePath);
        end
    end
end
clear i year
%figure;
trueImageBuddle = cell(1,6);
falseImageBuddle = cell(1,6);
for year = 1 : 6
    stretchedImage = cell(1,4);
    for i = 1 : 4
        imageDouble = double(bandSeries{year,i});
        minValue = prctile(imageDouble(:), 1);
        maxValue = prctile(imageDouble(:), 99);
        %maxValue = max(imageDouble,[],'all');
        stretchedImage{i} = uint16((imageDouble - minValue) / (maxValue - minValue) * 50000);
    end
    trueImageBuddle{year} = cat(3,stretchedImage{3},stretchedImage{2},stretchedImage{1});
    falseImageBuddle{year} = cat(3,stretchedImage{4},stretchedImage{3},stretchedImage{2});
end
clear year i
%save band.mat yearSeries bandSeries trueImageBuddle falseImageBuddle
%%
%load band.mat
global yearID;
yearID = ceil(rand() * 6);
global trueImage;
trueImage = trueImageBuddle{yearID};
global falseImage;
falseImage = falseImageBuddle{yearID};
global h;
h = imshow(trueImage);
mkdir('water');
mkdir('greenland');
mkdir('bareland');
mkdir('imprevious');
mkdir('cropland');
%% init UI
global xRange;
global yRange;
xRange = size(trueImage,2);
yRange = size(trueImage,1);
set(gcf,  'Position', [0 0 xRange yRange]);

global rectHandle;
rectHandle = [];
global counting;
counting.skip = 0;
counting.water = 0;
counting.greenland = 0;
counting.bareland = 0;
counting.imprevious = 0;
counting.cropland = 0;
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
global hCropland;
hCropland = uicontrol('Style', 'pushbutton', ...
                        'String', '这是耕地', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 6,baseY,buttonWidth,buttonHeight],...
                        'Callback', @croplandCallback);
global hCountCroplandLabel;
hCountCroplandLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 6, baseY - 15, 100, 20], ...
                        'String', ['耕地数:',num2str(counting.cropland)]);
global squareSize;
squareSize = 2;
global hSlider;
hSlider = uicontrol('Style', 'slider', ...
                        'Min', 1, ...
                        'Max', 20, ...
                        'Value', squareSize, ...
                        'Position', [baseX + gapWidth * 7, baseY + 5, 200, 20], ...
                        'Callback', @sliderCallback);
global hValueSizeLabel;
hValueSizeLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + gapWidth * 7, baseY - 15, 100, 20], ...
                        'String', ['当前滑块尺寸:',num2str(squareSize)]);

global data;
data = table('Size',[0,6],'VariableTypes',{'double','double','double','double','double','categorical'},'VariableNames',{'year','xmin','ymin','width','height','label'});

%% main function
moveToNext();
%xpos = squareSize+1;
%ypos = squareSize+1;
%xStart = xpos - squareSize;
%xEnd = xpos + squareSize;
%yStart = ypos - squareSize;
%yEnd = ypos + squareSize;
%rectHandle = rectangle('Position', [xStart, yStart, xEnd-xStart, yEnd-yStart], ...
%                       'EdgeColor', 'r', 'LineWidth', 2);
%position = get(rectHandle,'Position');
%disp(position);
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
    global yearID
    global yearSeries
    global bandSeries
    data = [data; table(yearSeries(yearID),position(1),position(2),position(3),position(4),categorical({'Water'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global counting;
    counting.water = counting.water+1;
    for i = 1 : 6
        imwrite(imcrop(bandSeries{yearID,i}, position),['water/',num2str(counting.water),'.tif'],"WriteMode","append");
    end
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
    global yearID     
    global yearSeries
    global bandSeries
    data = [data; table(yearSeries(yearID),position(1),position(2),position(3),position(4),categorical({'Greenland'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global counting;
    counting.greenland = counting.greenland+1;
    for i = 1 : 6
        imwrite(imcrop(bandSeries{yearID,i}, position),['greenland/',num2str(counting.greenland),'.tif'],"WriteMode","append");
    end
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
    global yearID  
    global yearSeries
    global bandSeries
    data = [data; table(yearSeries(yearID),position(1),position(2),position(3),position(4),categorical({'Bareland'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global counting;
    counting.bareland = counting.bareland+1;
    for i = 1 : 6
        imwrite(imcrop(bandSeries{yearID,i}, position),['bareland/',num2str(counting.bareland),'.tif'],"WriteMode","append");
    end
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
    global yearID   
    global yearSeries
    global bandSeries
    data = [data; table(yearSeries(yearID),position(1),position(2),position(3),position(4),categorical({'Imprevious'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global counting;
    counting.imprevious = counting.imprevious+1;
    for i = 1 : 6
        imwrite(imcrop(bandSeries{yearID,i}, position),['imprevious/',num2str(counting.imprevious),'.tif'],"WriteMode","append");
    end
    global hCountImpreviousLabel;
    set(hCountImpreviousLabel, 'String', ['建筑数:',num2str(counting.imprevious)]);
    moveToNext();
end
function croplandCallback(~, ~)
    global xRange;global yRange;
    set(gca, 'XLim', [0,xRange], 'YLim', [0,yRange]);
    global rectHandle
    position = get(rectHandle,'Position');
    %position(1) = position(1) - 13;
    %position(2) = position(2) - 13;
    global data
    global yearID  
    global yearSeries
    global bandSeries
    data = [data; table(yearSeries(yearID),position(1),position(2),position(3),position(4),categorical({'Cropland'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global counting;
    counting.cropland = counting.cropland+1;
    for i = 1 : 6
        imwrite(imcrop(bandSeries{yearID,i}, position),['cropland/',num2str(counting.cropland),'.tif'],"WriteMode","append");
    end
    global hCountCroplandLabel;
    set(hCountCroplandLabel, 'String', ['建筑数:',num2str(counting.cropland)]);
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
    global yearID;
    if (bandcompoent == true)
        global falseImageBuddle;
        set(h, 'CData', falseImageBuddle{yearID});
        bandcompoent = false;
    else
        global trueImageBuddle;
        set(h, 'CData', trueImageBuddle{yearID});
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
    global yearID
    yearID = ceil(rand() * 6);
    global h
    global bandcompoent
    global trueImageBuddle
    global falseImageBuddle
    if (bandcompoent)
        set(h, 'CData', trueImageBuddle{yearID});
    else
        set(h, 'CData', falseImageBuddle{yearID});
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