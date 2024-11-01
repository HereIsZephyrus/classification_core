#ifndef FUNC_HPP
#define FUNC_HPP
#include <opencv2/opencv.hpp>
#include <cstring>
#include <string>
#include <map>
#define SHOW_WINDOW false

typedef std::vector<float> vFloat;
typedef float** fMat;
namespace tcb{
bool TopHat(cv::Mat &image,int xSize,int ySize);
bool BottomHat(cv::Mat &image,int xSize,int ySize);
bool GaussianSmooth(cv::Mat &image,int xSize,int ySize,int sigma);
bool rgb2gray(cv::Mat &image);
bool Sobel(cv::Mat &image,int dx,int dy,int bandwidth);
bool Laplacian(cv::Mat &image,int bandwidth);
bool BoxSmooth(cv::Mat &image,int xSize,int ySize);
bool Erode(cv::Mat &image,int kernelSize);
bool Dilate(cv::Mat &image,int kernelSize);
bool drawCircleDDA(cv::Mat &image, int h, int k, float rx,float ry);
bool GenerateFeatureChannels(const cv::Mat &image,std::vector<cv::Mat> &channels);
bool CalcChannelMeanStds(const std::vector<cv::Mat> & channels, vFloat & data);
bool CalcInvMat(float** const convMat,float ** invMat,const int num);
bool CalcEigen(const std::vector<vFloat>& matrix, vFloat& eigVal, std::vector<vFloat>& eigVec, const int num);
}
namespace bayes{
double CalcConv(const vFloat& x, const vFloat& y);
}

#endif