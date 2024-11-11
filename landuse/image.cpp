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
    std::string SplitFolder = "../landuse/feature_city/split/";
    cv::imwrite(SplitFolder + "red.tif", bands[0]);
    cv::imwrite(SplitFolder + "green.tif", bands[1]);
    cv::imwrite(SplitFolder + "blue.tif", bands[2]);
    cv::imwrite(SplitFolder + "nir.tif", bands[3]);
    cv::Mat trueColor,falseColor;
    std::vector<cv::Mat> trueColorChannel = {bands[0],bands[1],bands[2]};
    std::vector<cv::Mat> falseColorChannel = {bands[3],bands[0],bands[1]};
    cv::merge(trueColorChannel,trueColor);
    cv::merge(falseColorChannel,falseColor);
    cv::imwrite(SplitFolder + "truecolor.tif", trueColor);
    cv::imwrite(SplitFolder + "falsecolor.tif", falseColor);
}
bool GenerateFeatureChannels(const cv::Mat &image,std::vector<cv::Mat> &channels){
    cv::split(image, channels);
    cv::Mat trueColor,falseColor;
    std::vector<cv::Mat> trueColorChannel = {channels[0],channels[1],channels[2]};
    std::vector<cv::Mat> falseColorChannel = {channels[1],channels[2],channels[3]};
    cv::merge(trueColorChannel,trueColor);
    cv::merge(falseColorChannel,falseColor);
    cv::Mat hsvTrueColor,hsvFalseColor;
    cv::cvtColor(trueColor, hsvTrueColor, cv::COLOR_BGR2HSV);
    cv::cvtColor(falseColor, hsvFalseColor, cv::COLOR_BGR2HSV);
    return true;
}
}