#include "image.hpp"
#include <filesystem>
#include <regex>
namespace fs = std::filesystem;
namespace weilaicheng{
void GenerateFeatureImage(cv::Mat& rawImage){
    cv::Mat image = cv::imread(FutureCityImage, cv::IMREAD_UNCHANGED);
    rawImage = image.clone();
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);
    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return;
    }
    std::vector<cv::Mat> bands;
    cv::split(image, bands);
    for (int i = 0; i < bands.size(); i++){
        double minVal, maxVal;
        cv::minMaxLoc(bands[i], &minVal, &maxVal);
        bands[i] = (bands[i] - minVal) * (255.0 / (maxVal - minVal));
        bands[i].convertTo(bands[i], CV_8U);
    }
    cv::merge(bands, rawImage);
    std::string SplitFolder = "../landuse/feature_city/split/";
    cv::imwrite(SplitFolder + "raw.tif", rawImage);
    cv::imwrite(SplitFolder + "blue.tif", bands[0]);
    cv::imwrite(SplitFolder + "green.tif", bands[1]);
    cv::imwrite(SplitFolder + "red.tif", bands[2]);
    cv::imwrite(SplitFolder + "nir.tif", bands[3]);
    cv::Mat trueColor,falseColor;
    std::vector<cv::Mat> trueColorChannel = {bands[2],bands[1],bands[0]};
    std::vector<cv::Mat> falseColorChannel = {bands[1],bands[2],bands[3]};
    cv::merge(trueColorChannel,trueColor);
    cv::merge(falseColorChannel,falseColor);
    cv::imwrite(SplitFolder + "truecolor.tif", trueColor);
    cv::imwrite(SplitFolder + "falsecolor.tif", falseColor);
    cv::Mat hsvFalseColor,hsvTrueColor;
    cv::cvtColor(falseColor, hsvFalseColor, cv::COLOR_BGR2HSV);
    cv::cvtColor(trueColor, hsvTrueColor, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvchannels;
    cv::split(hsvFalseColor, hsvchannels);
    bands.push_back(hsvchannels[0]);
    //hsvchannels.clear();
    //cv::split(hsvTrueColor, hsvchannels);
    //bands.push_back(hsvchannels[2]);
    cv::merge(bands, rawImage);
}
bool GenerateFeatureChannels(std::vector<cv::Mat> &channels){
    cv::Mat trueColor,falseColor;
    std::vector<cv::Mat> trueColorChannel = {channels[2],channels[1],channels[0]};
    std::vector<cv::Mat> falseColorChannel = {channels[1],channels[2],channels[3]};
    cv::merge(trueColorChannel,trueColor);
    cv::merge(falseColorChannel,falseColor);
    cv::Mat hsvFalseColor,hsvTrueColor;
    cv::cvtColor(trueColor, hsvTrueColor, cv::COLOR_BGR2HSV);
    cv::cvtColor(falseColor, hsvFalseColor, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvchannels;
    cv::split(hsvFalseColor, hsvchannels);
    channels.push_back(hsvchannels[0]);
    //hsvchannels.clear();
    //cv::split(hsvTrueColor, hsvchannels);
    //channels.push_back(hsvchannels[2]);
    return true;
}
}
namespace ningbo{
bool GenerateFeatureImage(int year,cv::Mat& featureImage,std::vector<float>& minVal,std::vector<float>& maxVal){
    const float itcps[6] = {0.3, 8.8, 6.1, 41.2, 25.4, 17.2};
    const float slopes[6] = {0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071};
    const int thresholdYear = 2015;
    const char signal[6] = {'1','2','3','4','5','7'};
    std::string folderPath = SeriesFolder + std::to_string(year) + "/";
    cv::Mat quality;
    std::vector<cv::Mat> bands;
    std::regex band_pattern(".*B.{1}\\.TIF", std::regex_constants::icase);
    std::regex QA_pattern(".*QA\\.TIF", std::regex_constants::icase);
    if (!fs::exists(folderPath)) {
        std::cerr << "Directory does not exist: " << folderPath << std::endl;
        return 1;
    }
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (std::regex_match(filename, band_pattern))
                bands.push_back(cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED));
            if (std::regex_match(filename, QA_pattern))
                quality = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        }
    }
    int row = quality.rows,col = quality.cols;
    cv::Mat cloudMask = cv::Mat::zeros(row,col,CV_8UC1);
    cv::Mat shadowMask = cv::Mat::zeros(row,col,CV_8UC1);
    for (int y = 0; y < row; y++){
        for (int x = 0; x < col; x++){
            uchar QAvalue = quality.at<uchar>(y,x);
            QAvalue /= 2;
            cloudMask.at<uchar>(y,x) = QAvalue%2;
            QAvalue /= 2;
            shadowMask.at<uchar>(y,x) = QAvalue%2;
        }
    }
    for (int i = 0; i < 6; i++){
        cv::Mat& band = bands[i];
        for (int y = 0; y < row; y++){
            for (int x = 0; x < col; x++){
                if (cloudMask.at<uchar>(y,x) == 1){
                    band.at<ushort>(y,x) = 65535;
                    continue;
                }
                if (shadowMask.at<uchar>(y,x) == 1)
                    band.at<ushort>(y,x) *= 1.1;
                if (year < thresholdYear && band.at<ushort>(y,x)>0)
                    band.at<ushort>(y,x) = band.at<ushort>(y,x) * slopes[i] + itcps[i];
                //if (band.at<ushort>(y,x))
                //    std::cout<<maxVal[i]<<' '<<minVal[i]<<' '<<band.at<ushort>(y,x)<<std::endl;
                //band.at<ushort>(y,x) = (band.at<ushort>(y,x) - minVal[i]) * (65535.0 / (maxVal[i] - minVal[i]));
            }
        }
    }
    cv::merge(bands,featureImage);
    return true;
}
char UrbanMaskAnalysis(const cv::Mat& lastImage,const cv::Mat& currentImage){
    char mainIncreaseDirection = 0;
    int row = lastImage.rows,col = lastImage.cols;
    std::vector<std::vector<int>> matrix(row,std::vector<int>(col,0));
    for (int y = 0; y < row; y++){
        for (int x = 0; x < col; x++){
            ushort lastValue = lastImage.at<ushort>(y,x);
            ushort currentValue = currentImage.at<ushort>(y,x);
            matrix[y][x] = currentValue - lastValue;
        }
    }
    int count[8];
    for (int i = 0; i < 8; i++)    count[i] = 0;
    for (int y = 0; y < row; y++)
        for (int x = 0; x < col; x++){
            int id = 0;
            if (x < col/2)    id += 4;
            if (y >= row/2)    id += 2;
            if (std::abs(x - col/2) / col < std::abs(y - row/2) / row)    id += 1;
            count[id] ++;
        }
    int maxCount = 0;
    for (int i = 0; i < 8; i++)
        if (count[i] > maxCount){
            maxCount = count[i];
            mainIncreaseDirection = i;
        }
    return mainIncreaseDirection;
}
}