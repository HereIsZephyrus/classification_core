#ifndef IMAGEHPP
#define IMAGEHPP
#define FutureCityImage "../landuse/feature_city/futureCity.tif"
#define SeriesFolder "../landuse/ningbo/data/"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
namespace weilaicheng{
void GenerateFeatureImage(cv::Mat& rawImage);
bool GenerateFeatureChannels(std::vector<cv::Mat> &channels);
}
namespace ningbo{
bool GenerateFeatureImage(int year,cv::Mat& featureImage,std::vector<float>& minVal,std::vector<float>& maxVal);
char UrbanMaskAnalysis(const cv::Mat& lastImage,const cv::Mat& currentImage);
}
#endif