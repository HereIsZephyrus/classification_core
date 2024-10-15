#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include "process.hpp"

bool FetchShadow(const cv::Mat &rawimage,cv::Mat &noShadowImage){
    std::vector<cv::Mat> channels;
    cv::split(noShadowImage, channels);
    cv::Mat redChannel = channels[2];
    cv::Mat greenChannel = channels[1];
    cv::Mat blueChannel = channels[0];
    cv::Mat image = blueChannel.clone();
    cv::Mat filteredImage = cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat kernelS = (cv::Mat_<uchar>(3, 3) << 
        5,-3,-3,
        5,0,-3,
        5,-3,-3);

    cv::Mat response;
    cv::filter2D(image, response, CV_32F, kernelS);
    cv::Mat maskImage;
    cv::convertScaleAbs(response, maskImage);
    maskImage = 255 - maskImage;
    for (int y = image.rows/2; y < image.rows; y++)
        for (int x = 0; x < image.cols; x++)
            if (redChannel.at<uchar>(y, x) - greenChannel.at<uchar>(y, x) > 20)
                maskImage.at<uchar>(y, x) = 0;
    if (SHOW_WINDOW){
        cv::imshow("Shadow mask", maskImage);
        cv::waitKey(0);
        cv::destroyWindow("Shadow mask");
    }
    tcb::Dilate(maskImage,40);
    tcb::Erode(maskImage,40);
    tcb::Dilate(maskImage,20);

    tcb::Erode(maskImage,20);
    tcb::Dilate(maskImage,20);
    if (SHOW_WINDOW){
        cv::imshow("Shadow mask", maskImage);
        cv::waitKey(0);
        cv::destroyWindow("Shadow mask");
    }
    cv::Mat shadowImage = maskImage.clone();
    tcb::Erode(shadowImage,15);
    tcb::Erode(shadowImage,15);
    tcb::Dilate(shadowImage,50);
    tcb::Dilate(shadowImage,50);
    tcb::Dilate(shadowImage,50);
    cv::Mat intersection;
    cv::bitwise_and(maskImage, shadowImage, intersection);
    if (SHOW_WINDOW){
        cv::imshow("Shadow fetch", intersection);
        cv::waitKey(0);
        cv::destroyWindow("Shadow fetch");
    }
    tcb::Erode(intersection,10);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
    cv::dilate(image, image, kernel);

    for (int y = 0; y < image.rows; y++)
        for (int x = 0; x < image.cols; x++){
            if (intersection.at<uchar>(y, x)){
                blueChannel.at<uchar>(y, x) += 70;
                redChannel.at<uchar>(y, x) += 70;
                greenChannel.at<uchar>(y, x) += 70;
            }
        }
    std::vector<cv::Mat> modifiedChannels;
    modifiedChannels.push_back(blueChannel);
    modifiedChannels.push_back(greenChannel);
    modifiedChannels.push_back(redChannel);
    cv::merge(modifiedChannels, noShadowImage);
    if (SHOW_WINDOW){
        cv::imshow("Decrease Shadow Image", noShadowImage);
        cv::waitKey(0);
        cv::destroyWindow("Decrease Shadow Image");
    }
    tcb::BoxSmooth(noShadowImage,30,30);
    return true;
}
bool CorrectImage(const cv::Mat& inputImage,cv::Mat& image){
    hist::vignetteCorrection(inputImage,image);
    tcb::GaussianSmooth(image,25,25,0);
    tcb::TopHat(image,20,20);
    tcb::GaussianSmooth(image,25,1,0);
    return true;
}

namespace hist{

bool SmoothImage(const cv::Mat& correctImage,cv::Mat& image) {
    cv::Mat templeteImage = image.clone();
    cv::Mat maskImage = correctImage.clone();
    tcb::Sobel(maskImage,0,1,11);
    fetchLines(maskImage);
    tcb::rgb2gray(maskImage);
    double threshold = cv::threshold(maskImage, maskImage, 254, 255, cv::THRESH_BINARY);
    if (SHOW_WINDOW){
        cv::imshow("tup Border Gray Sobel Image", maskImage);
        cv::waitKey(0);
        cv::destroyWindow("tup Border Gray Sobel Image");
    }
    tcb::Dilate(maskImage,9);
    tcb::Erode(maskImage,9);

    tcb::Erode(maskImage,9);
    tcb::Dilate(maskImage,9);
    
    tcb::Erode(maskImage,9);
    tcb::Erode(maskImage,9);
    int bandwidth = 40;
    for (int y = 0; y < maskImage.rows; y++)
        for (int x = 0; x < maskImage.cols; x++){
            if (maskImage.at<uchar>(y, x) == 0){
                int num = 0; 
                int average[3] = {0,0,0};
                for (int s = std::max(0,x - bandwidth/10); s < std::min(maskImage.cols - 1, x + bandwidth/10); s++)
                    for (int t = std::max(0,y - bandwidth); t < std::min(maskImage.rows - 1, y + bandwidth); t+=2)
                        if (((abs(s - x) > bandwidth / 2) || (abs(t - y) > bandwidth / 2)) &&maskImage.at<uchar>(t, s) > 0){
                            average[0] += cv::saturate_cast<uchar>(templeteImage.at<cv::Vec3b>(t, s)[0]);
                            average[1] += cv::saturate_cast<uchar>(templeteImage.at<cv::Vec3b>(t, s)[1]); 
                            average[2] += cv::saturate_cast<uchar>(templeteImage.at<cv::Vec3b>(t, s)[2]);
                            num++;
                        }
                if (num){
                    ////std::cout<<x<<' '<<y<<' '<<std::endl;
                    ////std::cout<<(int)templeteImage.at<cv::Vec3b>(y, x)[0]<<","<<(int)templeteImage.at<cv::Vec3b>(y, x)[1]<<","<<(int)templeteImage.at<cv::Vec3b>(y, x)[2]<<std::endl;
                    image.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>(average[0]/num);
                    image.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(average[1]/num);
                    image.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(average[2]/num);
                    ////std::cout<<(int)cv::saturate_cast<uchar>(average[0] / num)<<","<<(int)cv::saturate_cast<uchar>(average[1] / num)<<","<<(int)cv::saturate_cast<uchar>(average[2] / num)<<std::endl;
                    ////std::cout<<(int)image.at<cv::Vec3b>(y, x)[0]<<","<<(int)image.at<cv::Vec3b>(y, x)[1]<<","<<(int)image.at<cv::Vec3b>(y, x)[2]<<std::endl;
                }   
            }
        }
    tcb::BoxSmooth(image,15,15);
    tcb::BottomHat(image,40,40);
    tcb::GaussianSmooth(image,35,1,0);
    tcb::GaussianSmooth(image,49,49,0);
    tcb::Dilate(image,5);
    //cv::imwrite("flatImage.png", image);
    return true;
}
bool vignetteCorrection(const cv::Mat& inputImage, cv::Mat& convexImage) {
    CV_Assert(inputImage.type() == CV_8UC3);
    convexImage = inputImage.clone();
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    cv::Mat weight = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Point center(cols *2 / 5, rows / 2);
    float maxDist = std::sqrt(center.x * center.x + center.y * center.y);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float dist = std::sqrt((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y));
            weight.at<float>(y, x) = 0.7f + (dist / maxDist)/3.0f;
        }
    }
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x)
                convexImage.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(
                    inputImage.at<cv::Vec3b>(y, x)[c] * weight.at<float>(y, x)
                );
    if (SHOW_WINDOW){
        cv::imshow("vCorrection Image", convexImage);
        cv::waitKey(0);
        cv::destroyWindow("vCorrection Image");
    }
    return true;
}

bool fetchLines(cv::Mat &image){
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Mat redChannel = channels[2];
    cv::Mat greenChannel = channels[1];
    cv::Mat blueChannel = channels[0];
    for (int y = 0; y < image.rows; y++)
        for (int x = 0; x < image.cols; x++){
            if ((redChannel.at<uchar>(y, x) > 20) || (greenChannel.at<uchar>(y, x) > 20) || (blueChannel.at<uchar>(y, x) > 20)){
                blueChannel.at<uchar>(y, x) = 255;
                redChannel.at<uchar>(y, x) = 255;
                greenChannel.at<uchar>(y, x) = 255;
            }
        }
    std::vector<cv::Mat> modifiedChannels;
    modifiedChannels.push_back(blueChannel);
    modifiedChannels.push_back(greenChannel);
    modifiedChannels.push_back(redChannel);
    cv::merge(modifiedChannels, image);
    if (SHOW_WINDOW){
        cv::imshow("Line Detect Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Line Detect Image");
    }
    return true;
}
bool CreateMaskCr(const cv::Mat& rawImage, cv::Mat& totalMask) {
    cv::Mat testChannel = rawImage.clone();
    tcb::BoxSmooth(testChannel,70,70);
    tcb::Laplacian(testChannel,7);
    tcb::Dilate(testChannel,5);
    tcb::Erode(testChannel,11);
    double threshold = cv::threshold(testChannel, testChannel, 10, 255, cv::THRESH_BINARY);
    tcb::Erode(testChannel, 18);
    tcb::Erode(testChannel, 18);
    cv::Mat labels;
    int numComponents = cv::connectedComponents(testChannel, labels, 8);
    std::vector<Border> components;
    for (int i = 1; i < numComponents; i++) { 
        int size = cv::countNonZero(labels == i);
        if (size < 10000)
            continue;
        components.push_back(Border(i, size));
    }
    std::sort(components.begin(), components.end());
    for (int i = 0; i < 5; i++){
        cv::Mat mask;
        cv::compare(labels, components[i].label, mask, cv::CMP_EQ);
        cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
        cv::bitwise_or(totalMask, mask, totalMask);
    }
    if (SHOW_WINDOW){
        cv::imshow("Connected Component ", totalMask);
        cv::waitKey(0);
        cv::destroyWindow("Connected Component ");
    }
    return true;
}
bool CreateMaskCb(const cv::Mat& rawImage, cv::Mat& totalMask) {
    cv::Mat testChannel = rawImage.clone();
    tcb::BoxSmooth(testChannel,70,70);
    tcb::Laplacian(testChannel,7);
    tcb::Erode(testChannel,5);
    tcb::Dilate(testChannel,13);
    double threshold = cv::threshold(testChannel, testChannel, 10, 255, cv::THRESH_BINARY);
    tcb::Erode(testChannel,7);
    tcb::Erode(testChannel,7);
    tcb::Erode(testChannel, 18);
    tcb::Erode(testChannel, 18);
    cv::Mat labels;
    int numComponents = cv::connectedComponents(testChannel, labels, 8);
    std::vector<Border> components;
    for (int i = 1; i < numComponents; i++) { 
        int size = cv::countNonZero(labels == i);
        if (size < 10000)
            continue;
        components.push_back(Border(i, size));
    }
    std::sort(components.begin(), components.end());
    int remainLabel[3] = {0,10,11};
    for (int i = 0; i < 3; i++){
        int label = components[remainLabel[i]].label;
        cv::Mat mask;
        cv::compare(labels, label, mask, cv::CMP_EQ);
        cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
        //tcb::Erode(mask, 1);
        cv::bitwise_or(totalMask, mask, totalMask);
    }
    if (SHOW_WINDOW){
        cv::imshow("Connected Component ", totalMask);
        cv::waitKey(0);
        cv::destroyWindow("Connected Component ");
    }
    return true;
}

bool CreateMask(const cv::Mat& rawImage,cv::Mat& totalMask){
    cv::Mat ycrcbImage;
    cv::cvtColor(rawImage, ycrcbImage, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcbImage, channels);
    cv::Mat YChannel = channels[0];
    cv::Mat CrChannel = channels[1];
    cv::Mat CbChannel = channels[2];
    cv::Mat totalMaskCr = cv::Mat::zeros(rawImage.size(),  CV_8U),totalMaskCb = cv::Mat::zeros(rawImage.size(),  CV_8U);
    CreateMaskCb(CbChannel, totalMaskCb);
    CreateMaskCr(CrChannel, totalMaskCr);
    cv::bitwise_or(totalMaskCr, totalMaskCb, totalMask);
    if (SHOW_WINDOW){
        cv::imshow("Connected Component", totalMask);
        cv::waitKey(0);
        cv::destroyWindow("Connected Component");
    }
    return true;
}
bool FirstConvexHull(const cv::Mat& binaryImage,cv::Mat& convexImage){
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    convexImage = cv::Mat::zeros(binaryImage.size(), CV_8U);
    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull);
        std::vector<std::vector<cv::Point> > hullList;
        hullList.push_back(hull);
        cv::drawContours(convexImage,hullList, -1, 255, 2);
    }
    if (SHOW_WINDOW){
        cv::imshow("Convex Hulls", convexImage);
        cv::waitKey(0);
        cv::destroyWindow("Convex Hulls");
    }
    return true;
}
bool SecondConvexHull(const cv::Mat& convexIn,cv::Mat& totalMask,cv::Mat& convexOut){
    cv::Mat labels;
    int numComponents = cv::connectedComponents(convexIn, labels, 8);
    std::vector<Border> borders;
    for (int i = 1; i < numComponents; i++) {
        int size = cv::countNonZero(labels == i);
        cv::Mat mask;
        cv::compare(labels, i, mask, cv::CMP_EQ);
        cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
        borders.push_back(Border(i, size));
        //std::cout<<borders[i-1].count<<std::endl;
        int ytopx = 0,ytopy = 0,xleftx = 10000,xlefty = 0;
        for(int x = 0; x < mask.rows; x++)
            for (int y = 0; y < mask.cols; y++)
                if(mask.at<uchar>(x,y)){
                    borders[i-1].centerX += x;
                    borders[i-1].centerY += y;
                    if (y>ytopy){
                        ytopy = y;
                        ytopx = x;
                    }
                    if (x<xleftx){
                        xleftx = x;
                        xlefty = y;
                    }
                }
        borders[i-1].centerX /= borders[i-1].count;
        borders[i-1].centerY /= borders[i-1].count;
        float radiusy = std::sqrt((borders[i-1].centerX - ytopx) * (borders[i-1].centerX - ytopx) + (borders[i-1].centerY - ytopy) * (borders[i-1].centerY - ytopy)) - 35;
        float radiusx = std::sqrt((borders[i-1].centerX - xleftx) * (borders[i-1].centerX - xleftx) + (borders[i-1].centerY - xlefty) * (borders[i-1].centerY - xlefty)) - 35;
        tcb::drawCircleDDA(totalMask, borders[i-1].centerY, borders[i-1].centerX, radiusx,radiusy);
    }
    tcb::Dilate(totalMask, 5);
    tcb::Erode(totalMask, 5);
    if (SHOW_WINDOW){
        cv::imshow("Border Mask", totalMask);
        cv::waitKey(0);
        cv::destroyWindow("Border Mask");
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(totalMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    convexOut = cv::Mat::zeros(totalMask.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> hull;
        cv::convexHull(contours[i], hull);
        cv::fillConvexPoly(convexOut, hull, cv::Scalar(255, 255, 255));
    }
    if (SHOW_WINDOW){
        cv::imshow("Convex Hulls", convexOut);
        cv::waitKey(0);
        cv::destroyWindow("Convex Hulls");
    }
    return true;
}
bool ClassifityFruits(const cv::Mat& rawImage,const cv::Mat& correctImage,const cv::Mat& flatImage,const cv::Mat& convexImage){
    cv::Mat labels;
    cv::Mat secondConvexImage = convexImage.clone();
    tcb::rgb2gray(secondConvexImage);
    int numComponents = cv::connectedComponents(secondConvexImage, labels, 8);
    std::vector<Border> components;
    for (int i = 1; i < numComponents; i++) { 
        cv::Mat mask;
        cv::compare(labels, i, mask, cv::CMP_EQ);
        tcb::Erode(mask,5);
        cv::Rect bounding = cv::boundingRect(mask);
        cv::Mat croppedImage = flatImage(bounding),croppedMask = mask(bounding),croppedRawImage = rawImage(bounding);
        cv::Mat smoothFruit,fruit;
        std::vector<int> maskPixels;
        cv::cvtColor(croppedImage, smoothFruit, cv::COLOR_BGR2BGRA);
        cv::cvtColor(croppedRawImage, fruit, cv::COLOR_BGR2BGRA);
        int maxp = 0,bandwidth = 5;
        double total = 0;
        for (int x = 0; x < croppedMask.cols - bandwidth; x++)
            for (int y = 0; y < croppedMask.rows - bandwidth; y++) 
                if (croppedMask.at<uchar>(y,x) == 0){
                    smoothFruit.at<cv::Vec4b>(y,x)[3] = 0;
                    fruit.at<cv::Vec4b>(y,x)[3] = 0;
                }
                else{
                    int res = 0,count = 0;
                    for (int s = 0; s < bandwidth; s++)
                        for (int t = 0; t < bandwidth; t++){
                            if (croppedMask.at<uchar>(y+s,x+t)){
                                res +=  correctImage.at<cv::Vec3b>(y+s,x+t)[0]+correctImage.at<cv::Vec3b>(y+s,x+t)[1]+correctImage.at<cv::Vec3b>(y+s,x+t)[2];
                                count ++;
                            }
                        }
                    res/=count;
                    total+=res;
                    maxp = std::max(maxp,res);
                    maskPixels.push_back(res);
                }
                    
        cv::imwrite("../results/" + std::to_string(i) + "-temp.png", smoothFruit);
        double mean = static_cast<double>(total/maskPixels.size()),variance = 0;
        for (std::vector<int>::iterator it = maskPixels.begin(); it != maskPixels.end(); it++)
            variance += (*it - mean) * (*it - mean);
        variance = variance / maskPixels.size();
        std::cout<<i<<' '<<std::sqrt(variance)<<' '<<mean<<' '<<maxp<<std::endl;
        if (variance < 22 * 22){
            if (mean < 130)
                cv::imwrite("../results/" + std::to_string(i) + "-blackplum.png", fruit);
        }else if (mean < 295)
            cv::imwrite("../results/" + std::to_string(i) + "-grape.png", fruit);
        else if (mean < 300)
            cv::imwrite("../results/" + std::to_string(i) + "-dongzao.png", fruit);
        else if (mean < 320)
            cv::imwrite("../results/" + std::to_string(i) + "-apple.png", fruit);
        else if (maxp>430)
            cv::imwrite("../results/" + std::to_string(i) + "-peach.png", fruit);
        else
            cv::imwrite("../results/" + std::to_string(i) + "-yellowpeach.png", fruit);
    }
    return true;
}
};
namespace bayes{
bool CalcClassProb(float* prob){
    unsigned int* countings = new unsigned int[Classes::counter];
    unsigned int totalRecord = 0;
    for (int i = 0; i < Classes::counter; i++)
        countings[i] = 0;
    std::string filename = "../sampling/suit3/classification.csv";
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "can't open file!" << filename << std::endl;
        return false;
    }
    std::getline(file, line);// throw header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;
        while (std::getline(ss, value, ',')) {}
        totalRecord++;
        value.pop_back();
        if (value == "Desk")
            countings[Classes::Desk]++;
        else if (value == "Apple")
            countings[Classes::Apple]++;
        else if (value == "Blackplum")
            countings[Classes::Blackplum]++;
        else if (value == "Dongzao")
            countings[Classes::Dongzao]++;
        else if (value == "Grape")
            countings[Classes::Grape]++;
        else if (value == "Peach")
            countings[Classes::Peach]++;
        else if (value == "Yellowpeach")
            countings[Classes::Yellowpeach]++;
    }
    file.close();
    for (int i = 0; i < Classes::counter; i++)
        prob[i] = static_cast<float>(countings[i]) / totalRecord;
    delete[] countings;
    return true;
}
bool StudySamples(StaticPara* classParas){
    namespace fs = std::filesystem;
    for (unsigned int suitID = 0; suitID < 3; suitID++){// has sampled 3 suit of samples
        std::string suitFolderPath = "../sampling/suit" + std::to_string(suitID + 1) + "/";
        for (unsigned int classID = 0; classID < Classes::counter; classID++){
            std::string classFolderPath = suitFolderPath + classFolderNames[classID] + "/";
            if (!fs::exists(classFolderPath))
                continue;
            for (const auto& entry : fs::recursive_directory_iterator(classFolderPath)) {
                classParas[classID].Sampling(entry.path());
            }
        }
    }
    for (unsigned int classID = 0; classID < Classes::counter; classID++)
        classParas[classID].printInfo();
    return true;
}
bool BayesClassify(const cv::Mat& rawimage,NaiveBayesClassifier* classifer,std::vector<std::vector<Classes>>& patchClasses){
    int rows = rawimage.rows, cols = rawimage.cols;
    for (int r = classifierKernelSize/2; r <= rows - classifierKernelSize; r+=classifierKernelSize/2){
        std::vector<Classes> rowClasses;
        bool lastRowCheck = (r >= (rows - classifierKernelSize));
        for (int c = classifierKernelSize/2; c <= cols - classifierKernelSize; c+=classifierKernelSize/2){
            bool lastColCheck = (c >= (cols - classifierKernelSize));
            cv::Rect window(c - classifierKernelSize/2 ,r - classifierKernelSize/2,  classifierKernelSize - lastColCheck, classifierKernelSize - lastRowCheck);  
            cv::Mat sample = rawimage(window);
            std::vector<cv::Mat> channels;
            vFloat means;
            tcb::GenerateFeatureChannels(sample, channels);
            tcb::CalcChannelMeans(channels, means);
            rowClasses.push_back(classifer->Predict(means));
        }
        patchClasses.push_back(rowClasses);
    }
    return true;
}
bool DownSampling(const ClassMat& patchClasses,ClassMat& pixelClasses){
    ClassMat::const_iterator row = patchClasses.begin();
    { //tackle the first line
        vClasses temprow;
        for (vClasses::const_iterator col = row->begin(); col != row->end(); col++){
            if (col == row->begin()){
                temprow.push_back(*col);
                continue;
            }
            if ((*col) != (*(col-1)))//horizontalEdgeCheck
                temprow.push_back(Classes::Edge);
            else
                temprow.push_back(*col);
            if ((col+1) == row->end())// manually add the last element
                temprow.push_back(*col);
        }
        pixelClasses.push_back(temprow);
    }
    vClasses::const_iterator lastRowBegin = row->begin();
    row++;
    for (; row != patchClasses.end(); row++){
        vClasses temprow;
        vClasses::const_iterator col = row->begin(),diagonalCol = lastRowBegin;
        { //tackle the first col
            if (*col == *diagonalCol)
                temprow.push_back(*col);
            else
                temprow.push_back(Classes::Edge);
            col++;
        }
        for (;col != row->end(); col++,diagonalCol++){
            bool horizontalEdgeCheck = (*col) == (*(col-1));
            bool verticalEdgeCheck = (*col) == (*(diagonalCol+1));
            bool diagonalEdgeCheck = (*col) == (*(diagonalCol));
            if (horizontalEdgeCheck && verticalEdgeCheck && diagonalEdgeCheck)
                temprow.push_back(*col);
            else
                temprow.push_back(Classes::Edge);
            if ((col+1) == row->end())// manually add the last element
                temprow.push_back(*col);
        }
        pixelClasses.push_back(temprow);
        if (row+1 != patchClasses.end())//pause on the last row
            lastRowBegin = row->begin();
    }
    row--;
    {// manually add the last row
        vClasses temprow;
        vClasses::const_iterator col = row->begin(),diagonalCol = lastRowBegin;
        { //tackle the first col
            if (*col == *diagonalCol)
                temprow.push_back(*col);
            else
                temprow.push_back(Classes::Edge);
            col++;
        }
        for (;col != row->end(); col++,diagonalCol++){
            bool horizontalEdgeCheck = (*col) == (*(col-1));
            bool verticalEdgeCheck = (*col) == (*(diagonalCol+1));
            bool diagonalEdgeCheck = (*col) == (*(diagonalCol));
            if (horizontalEdgeCheck && verticalEdgeCheck && diagonalEdgeCheck)
                temprow.push_back(*col);
            else
                temprow.push_back(Classes::Edge);
            if ((col+1) == row->end())// manually add the last element
                temprow.push_back(*col);
        }
        pixelClasses.push_back(temprow);
    }
    return true;
}
bool GenerateClassifiedImage(const cv::Mat& rawimage,cv::Mat& classified,const ClassMat& pixelClasses){
    classified = cv::Mat::zeros(rawimage.rows, rawimage.cols, CV_8UC3);
    classified.setTo(classifyColor[Unknown]);
    int y = 0;
    for (ClassMat::const_iterator row = pixelClasses.begin(); row != pixelClasses.end(); row++,y+=classifierKernelSize/2){
        int x = 0;
        for (vClasses::const_iterator col = row->begin(); col != row->end(); col++,x+=classifierKernelSize/2){
            if (x >= rawimage.cols - classifierKernelSize/2)
                break;
            cv::Rect window(x,y,classifierKernelSize/2,classifierKernelSize/2);
            classified(window) = classifyColor[*col];
        }
        if (y >= rawimage.rows - classifierKernelSize/2)
            break;
    }
    return true;
}
};