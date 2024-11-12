#include "image.hpp"
namespace weilaicheng{
void generateFeatureImage(cv::Mat& rawImage){
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