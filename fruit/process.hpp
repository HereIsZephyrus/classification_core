#ifndef PROCESSHPP
#define PROCESSHPP
#include <opencv2/opencv.hpp>
#include "fruit.hpp"
#include "classifier.hpp"

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
template <class paraForm>
bool BayesClassify(const cv::Mat& rawimage,T_BayesClassifier<paraForm,Classes>* classifer,std::vector<std::vector<Classes>>& patchClasses){
    int rows = rawimage.rows, cols = rawimage.cols;
    for (int r = classifierKernelSize/2; r <= rows - classifierKernelSize; r+=classifierKernelSize/2){
        std::vector<Classes> rowClasses;
        bool lastRowCheck = (r >= (rows - classifierKernelSize));
        for (int c = classifierKernelSize/2; c <= cols - classifierKernelSize; c+=classifierKernelSize/2){
            bool lastColCheck = (c >= (cols - classifierKernelSize));
            cv::Rect window(c - classifierKernelSize/2 ,r - classifierKernelSize/2,  classifierKernelSize - lastColCheck, classifierKernelSize - lastRowCheck);  
            cv::Mat sample = rawimage(window);
            std::vector<cv::Mat> channels;
            vFloat data;
            tcb::GenerateFeatureChannels(sample, channels);
            tcb::CalcChannelMeanStds(channels, data);
            rowClasses.push_back(classifer->Predict(data));
        }
        patchClasses.push_back(rowClasses);
    }
    return true;
}
bool DownSampling(const ClassMat& patchClasses,ClassMat& pixelClasses);
bool GenerateClassifiedImage(const cv::Mat& rawimage,cv::Mat& classified,const ClassMat& pixelClasses);
};
#endif