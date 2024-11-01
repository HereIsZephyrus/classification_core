#include "image.hpp"
void generateFeatureImage(cv::Mat& rawImage){
    cv::Mat image = cv::imread(FutureCityImage, cv::IMREAD_UNCHANGED);
    rawImage = image.clone();
    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return;
    }
    std::vector<cv::Mat> bands;
    cv::split(image, bands);
    std::string SplitFolder = "../landuse/feature_city/split/";
    cv::imwrite(SplitFolder + "blue.tif", bands[0]);
    cv::imwrite(SplitFolder + "green.tif", bands[1]);
    cv::imwrite(SplitFolder + "red.tif", bands[2]);
    cv::imwrite(SplitFolder + "nir.tif", bands[3]);
    cv::Mat trueColor,falseColor;
    std::vector<cv::Mat> trueColorChannel = {bands[0],bands[1],bands[2]};
    std::vector<cv::Mat> falseColorChannel = {bands[1],bands[2],bands[3]};
    cv::merge(trueColorChannel,trueColor);
    cv::merge(falseColorChannel,falseColor);
    cv::imwrite(SplitFolder + "truecolor.tif", trueColor);
    cv::imwrite(SplitFolder + "falsecolor.tif", falseColor);
}