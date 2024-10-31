clc;clear;close all;
%% init image and interface
global rawImage;
rawImage = imread("raw.jpg");
figure;
imshow(rawImage);
global autolabel;
autolabel = true;
mkdir('apple');
mkdir('blackplum');
mkdir('peach');
mkdir('yellowpeach');
mkdir('grape');
mkdir('dongzao');
mkdir('desk');
%% init UI
global xRange;
global yRange;
xRange = size(rawImage,2);
yRange = size(rawImage,1);
set(gcf,  'Position', [0 0 xRange/2 yRange/2]);

global rectHandle;
rectHandle = [];
global counting;
counting.apple = 0;
counting.blackplum = 0;
counting.peach = 0;
counting.yellowpeach = 0;
counting.dongzao = 0;
counting.grape = 0;
counting.desk = 0;
baseX = 70; baseY = 20; buttonWidth = 85; buttonHeight = 30; gapWidth = 100;
hFinish = uicontrol('Style', 'pushbutton', ...
                        'String', '结束选择', ...
                        'FontSize',14, ...
                        'Position',[baseX,baseY,buttonWidth,buttonHeight],...
                        'Callback', @finishCallback);
set(hFinish,'Visible','on');
global hApple;
hApple = uicontrol('Style', 'pushbutton', ...
                            'String', '这是苹果', ...
                            'FontSize',14, ...
                            'Position',[baseX + gapWidth * 1,baseY,buttonWidth,buttonHeight],...
                            'Callback', @appleCallback);
set(hApple,'Visible','off');
global hBlackplum;
hBlackplum = uicontrol('Style', 'pushbutton', ...
                        'String', '这是黑李', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 2,baseY,buttonWidth,buttonHeight],...
                        'Callback', @blackplumCallback);
set(hBlackplum,'Visible','off');
global hPeach;
hPeach = uicontrol('Style', 'pushbutton', ...
                        'String', '这是桃子', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 3,baseY,buttonWidth,buttonHeight],...
                        'Callback', @peachCallback);
set(hPeach,'Visible','off');
global hYellowpeach;
hYellowpeach = uicontrol('Style', 'pushbutton', ...
                        'String', '这是黄桃', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 4,baseY,buttonWidth,buttonHeight],...
                        'Callback', @yellowpeachCallback);
set(hYellowpeach,'Visible','off');
global hDongzao;
hDongzao = uicontrol('Style', 'pushbutton', ...
                        'String', '这是冬枣', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 5,baseY,buttonWidth,buttonHeight],...
                        'Callback', @dongzaoCallback);
set(hDongzao,'Visible','off');
global hGrape;
hGrape = uicontrol('Style', 'pushbutton', ...
                        'String', '这是葡萄', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 6,baseY,buttonWidth,buttonHeight],...
                        'Callback', @grapeCallback);
set(hGrape,'Visible','off');
global hDesk;
hDesk = uicontrol('Style', 'pushbutton', ...
                        'String', '这是桌子', ...
                        'FontSize',14, ...
                        'Position',[baseX + gapWidth * 7,baseY,buttonWidth,buttonHeight],...
                        'Callback', @deskCallback);
set(hDesk,'Visible','off');

global squareSize;
squareSize = 35;
global hSlider;
hSlider = uicontrol('Style', 'slider', ...
                        'Min', 50, ...
                        'Max', 400, ...
                        'Value', squareSize, ...
                        'Position', [baseX + 30, 120, 200, 20], ...
                        'Callback', @sliderCallback);
global hValueLabel;
hValueLabel = uicontrol('Style', 'text', ...
                        'Position', [baseX + 30, 100, 100, 20], ...
                        'String', ['当前滑块尺寸:',num2str(squareSize)]);

global data;
data = table('Size',[0,5],'VariableTypes',{'double','double','double','double','categorical'},'VariableNames',{'xmin','ymin','width','height','label'});

%% main function
if (autolabel)
    global xpos;
    xpos = 0;
    global ypos;
    ypos = 0;
    global hSkip;
    hSkip = uicontrol('Style', 'pushbutton', ...
                            'String', '跳过样本', ...
                            'FontSize',14, ...
                            'Position',[baseX + gapWidth * 8,baseY,buttonWidth,buttonHeight],...
                            'Callback', @skipCallback);
    set(hSkip,'Visible','off');
    global hStart;
    hStart = uicontrol('Style', 'pushbutton', ...
                            'String', '开始采样', ...
                            'FontSize',14, ...
                            'Position',[baseX + gapWidth * 1,baseY,buttonWidth,buttonHeight],...
                            'Callback', @startCallback);
    
else
    set(gcf, 'WindowButtonDownFcn', @createSquare);
end

%% functions
function createSquare(~, ~)
    % get cursor position
    pt = get(gca, 'CurrentPoint');
    xClick = round(pt(1, 1));
    yClick = round(pt(1, 2));

    % get boundary
    global squareSize;
    global xRange;
    global yRange;
    xStart = xClick - squareSize;
    xEnd = xClick + squareSize;
    yStart = yClick - squareSize;
    yEnd = yClick + squareSize;
    global rectHandle;
    global hApple;
    global hBlackplum;
    global hPeach;
    global hYellowpeach;
    global hDongzao;
    global hGrape;
    global hDesk;
    if (xEnd > xRange || xStart < 1 || yEnd > yRange || yStart < 1)
        set(rectHandle,'Visible','off');
        set(hApple,'Visible','off');
        set(hBlackplum,'Visible','off');
        set(hPeach,'Visible','off');
        set(hYellowpeach,'Visible','off');
        set(hDongzao,'Visible','off');
        set(hGrape,'Visible','off');
        set(hDesk,'Visible','off');
        return
    end
    if ~isempty(rectHandle)
        delete(rectHandle);
    end
    hold on;
    rectHandle = rectangle('Position', [xStart, yStart, xEnd-xStart, yEnd-yStart], ...
                           'EdgeColor', 'r', 'LineWidth', 2);
    set(rectHandle,'Visible','on');
    set(hApple,'Visible','on');
    set(hBlackplum,'Visible','on');
    set(hPeach,'Visible','on');
    set(hYellowpeach,'Visible','on');
    set(hDongzao,'Visible','on');
    set(hGrape,'Visible','on');
    set(hDesk,'Visible','on');
    hold off;
end


function finishCallback(~, ~)
    global data
    writetable(data,'classification.csv');
    close all
end
function appleCallback(~, ~)
    global autolabel
    global rectHandle
    positions = get(rectHandle,'Position');
    positions(1) = positions(1) - 13;
    positions(2) = positions(2) - 13;
    global data
    data = [data; table(positions(1),positions(2),positions(3),positions(4),categorical({'Apple'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    patch = imcrop(rawImage, positions);
    global counting;
    counting.apple = counting.apple+1;
    imwrite(patch,['apple/',num2str(counting.apple),'.png']);
    if (autolabel)
        moveToNext();
    end
end
function blackplumCallback(~, ~)
    global autolabel
    global rectHandle
    positions = get(rectHandle,'Position');
    positions(1) = positions(1) - 13;
    positions(2) = positions(2) - 13;
    global data
    data = [data; table(positions(1),positions(2),positions(3),positions(4),categorical({'Blackplum'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    patch = imcrop(rawImage, positions);
    global counting;
    counting.blackplum = counting.blackplum+1;
    imwrite(patch,['blackplum/',num2str(counting.blackplum),'.png']);
    if (autolabel)
        moveToNext();
    end
end
function peachCallback(~, ~)
    global autolabel
    global rectHandle
    positions = get(rectHandle,'Position');
    positions(1) = positions(1) - 13;
    positions(2) = positions(2) - 13;
    global data
    data = [data; table(positions(1),positions(2),positions(3),positions(4),categorical({'Peach'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    patch = imcrop(rawImage, positions);
    global counting;
    counting.peach = counting.peach+1;
    imwrite(patch,['peach/',num2str(counting.peach),'.png']);
    if (autolabel)
        moveToNext();
    end
end
function yellowpeachCallback(~, ~)
    global rectHandle
    positions = get(rectHandle,'Position');
    positions(1) = positions(1) - 13;
    positions(2) = positions(2) - 13;
    global data
    data = [data; table(positions(1),positions(2),positions(3),positions(4),categorical({'Yellowpeach'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    patch = imcrop(rawImage, positions);
    global counting;
    counting.yellowpeach = counting.yellowpeach+1;
    imwrite(patch,['yellowpeach/',num2str(counting.yellowpeach),'.png']);
    global autolabel
    if (autolabel)
        moveToNext();
    end
end
function dongzaoCallback(~, ~)
    global rectHandle
    positions = get(rectHandle,'Position');
    positions(1) = positions(1) - 13;
    positions(2) = positions(2) - 13;
    global data
    data = [data; table(positions(1),positions(2),positions(3),positions(4),categorical({'Dongzao'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    patch = imcrop(rawImage, positions);
    global counting;
    counting.dongzao = counting.dongzao+1;
    imwrite(patch,['dongzao/',num2str(counting.dongzao),'.png']);
    global autolabel
    if (autolabel)
        moveToNext();
    end
end
function grapeCallback(~, ~)
    global rectHandle
    positions = get(rectHandle,'Position');
    positions(1) = positions(1) - 13;
    positions(2) = positions(2) - 13;
    global data
    data = [data; table(positions(1),positions(2),positions(3),positions(4),categorical({'Grape'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    patch = imcrop(rawImage, positions);
    global counting;
    counting.grape = counting.grape+1;
    imwrite(patch,['grape/',num2str(counting.grape),'.png']);
    global autolabel
    if (autolabel)
        moveToNext();
    end
end
function deskCallback(~, ~)
    global rectHandle
    positions = get(rectHandle,'Position');
    positions(1) = positions(1) - 13;
    positions(2) = positions(2) - 13;
    global data
    data = [data; table(positions(1),positions(2),positions(3),positions(4),categorical({'Desk'}),'VariableNames',data.Properties.VariableNames)];
    %disp(data);
    global rawImage;
    patch = imcrop(rawImage, positions);
    global counting;
    counting.desk = counting.desk+1;
    imwrite(patch,['desk/',num2str(counting.desk),'.png']);
    global autolabel
    if (autolabel)
        moveToNext();
    end
end
function skipCallback(~, ~)
    moveToNext();
end
function startCallback(~, ~)
    global squareSize;
    global xpos;global ypos;
    global rectHandle;
    xpos = squareSize + 30;
    ypos = squareSize + 30;
    xStart = xpos - squareSize;
    xEnd = xpos + squareSize;
    yStart = ypos - squareSize;
    yEnd = ypos + squareSize;
    hold on;
    rectHandle = rectangle('Position', [xStart, yStart, xEnd-xStart, yEnd-yStart], ...
                           'EdgeColor', 'r', 'LineWidth', 2);
    hold off;
    set(rectHandle,'Visible','on');
    global hApple;
    global hBlackplum;
    global hPeach;
    global hYellowpeach;
    global hDongzao;
    global hGrape;
    global hDesk;
    global hSkip;
    global hStart;
    set(hApple,'Visible','on');
    set(hBlackplum,'Visible','on');
    set(hPeach,'Visible','on');
    set(hYellowpeach,'Visible','on');
    set(hDongzao,'Visible','on');
    set(hGrape,'Visible','on');
    set(hDesk,'Visible','on');
    set(hSkip,'Visible','on');
    set(hStart,'Visible','off');
end

function sliderCallback(~, ~)
    global hSlider;
    global squareSize;
    global hValueLabel;
    sliderValue = get(hSlider, 'Value');
    squareSize = round(sliderValue);
    set(hValueLabel, 'String', ['当前值:',num2str(squareSize)]);
end

function moveToNext()
    global squareSize;
    global xpos;global ypos;
    global rectHandle;
    xpos = xpos + squareSize * 5;
    global xRange
    if (xpos > xRange - (squareSize + 10))
        xpos = squareSize + 30;
        ypos = ypos + squareSize * 5;
        global yRange
        if (ypos > yRange - (squareSize + 10))
            ypos = ypos - squareSize * 5;
        end
    end
    xStart = xpos - squareSize;
    xEnd = xpos + squareSize;
    yStart = ypos - squareSize;
    yEnd = ypos + squareSize;
    hold on;
    rectHandle = rectangle('Position', [xStart, yStart, xEnd-xStart, yEnd-yStart], ...
                           'EdgeColor', 'r', 'LineWidth', 2);
    hold off;
end