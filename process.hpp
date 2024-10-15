#ifndef PROCESSHPP
#define PROCESSHPP
#include <opencv2/opencv.hpp>
#include "func.hpp"

bool FetchShadow(const cv::Mat &rawimage,cv::Mat &noShadowImage);
bool CorrectImage(const cv::Mat& inputImage,cv::Mat& image);
namespace hist{
bool SmoothImage(const cv::Mat& correctImage,cv::Mat& image);
bool vignetteCorrection(const cv::Mat& inputImage, cv::Mat& convexImage);
bool fetchLines(cv::Mat &image);
bool CreateMaskCr(const cv::Mat& rawImage, cv::Mat& totalMask);
bool CreateMaskCb(const cv::Mat& rawImage, cv::Mat& totalMask);
bool CreateMask(const cv::Mat& rawImage,cv::Mat& totalMask);
bool FirstConvexHull(const cv::Mat& binaryImage,cv::Mat& convexImage);
bool SecondConvexHull(const cv::Mat& convexIn,cv::Mat& totalMask,cv::Mat& convexOut);
bool ClassifityFruits(const cv::Mat& rawImage,const cv::Mat& correctImage,const cv::Mat& flatImage,const cv::Mat& convexImage);
};
typedef std::vector<std::vector<Classes>> ClassMat;
namespace bayes{
bool CalcClassProb(float* prob);
bool StudySamples(StaticPara* classParas,std::vector<Sample>& dataset);
bool BayesClassify(const cv::Mat& rawimage,NaiveBayesClassifier* classifer,std::vector<std::vector<Classes>>& patchClasses);
bool DownSampling(const ClassMat& patchClasses,ClassMat& pixelClasses);
bool GenerateClassifiedImage(const cv::Mat& rawimage,cv::Mat& classified,const ClassMat& pixelClasses);
};
#endif