#ifndef IMAGEHPP
#define IMAGEHPP
#define FutureCityImage "../landuse/feature_city/futureCity.tif"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
namespace weilaicheng{
void GenerateFeatureImage(cv::Mat& rawImage);
bool GenerateFeatureChannels(std::vector<cv::Mat> &channels);
}
namespace ningbo{
bool GenerateFeatureImage(int year,cv::Mat& featureImage);
bool ReadRawImage(int year,cv::Mat& rawImage,std::vector<float>& MINVAL,std::vector<float>& MAXVAL);
char UrbanMaskAnalysis(std::shared_ptr<cv::Mat> lastImage,std::shared_ptr<cv::Mat> currentImage);
}
#endif