#ifndef IMAGEHPP
#define IMAGEHPP
#define FutureCityImage "../landuse/feature_city/futureCity.tif"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
namespace weilaicheng{
void generateFeatureImage(cv::Mat& rawImage);
bool GenerateFeatureChannels(std::vector<cv::Mat> &channels);
}
#endif