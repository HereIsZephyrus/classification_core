#ifndef FRUITHPP
#define FRUITHPP
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../func.hpp"
#include "process.hpp"

struct Border{
    int label;
    int count;
    long long centerX,centerY;
    bool operator <(const Border &b) const {
        return count > b.count;
    }
    Border(int label,int count){
        this->label = label;
        this->count = count;
        this->centerX = -1;
        this->centerY = -1;
    }
};
enum Classes : unsigned int{
    Desk,
    Apple,
    Blackplum,
    Dongzao,
    Grape,
    Peach,
    Yellowpeach,
    counter,
    Edge,
    Unknown
};
extern std::string classFolderNames[Classes::counter];
extern std::unordered_map<Classes,cv::Scalar> classifyColor;
enum Demisions : unsigned int{
    hue,
    saturation,
    value,
    //gradient,
    angle,
    dim
};
typedef std::vector<Classes> vClasses;
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

int FruitMain();
void PreProcess(const cv::Mat& rawImage, cv::Mat&processed);
int HistMethod(const cv::Mat& rawImage);
int BayersMethod(const cv::Mat& correctImage);
int FisherMethod(const cv::Mat& correctImage);
#endif