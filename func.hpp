#ifndef FUNC_HPP
#define FUNC_HPP
#include <opencv2/opencv.hpp>
#include <cstring>
#include <string>
#include <map>
#define SHOW_WINDOW false
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
extern std::map<Classes,cv::Scalar> classifyColor;
enum Demisions : unsigned int{
    blue,
    green,
    red,
    gray,
    gradient,
    angle,
    dim
};
constexpr int classifierKernelSize = 9;
typedef std::vector<float>::iterator pfloat;
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
};
namespace bayes{
class StaticPara{
    std::vector<std::vector<float>> mu,sigma;
    Classes classID;
    unsigned int recordNum;
public:
    StaticPara() = default;
    StaticPara(Classes classID):recordNum(0),classID(classID){
        mu.reserve(Demisions::dim);
        sigma.reserve(Demisions::dim);
    }
    ~StaticPara(){}
    void InitClassType(Classes ID);
    void Sampling(const std::string& entryPath);
    float combineMu(pfloat begin,pfloat end);
    float combineSigma(pfloat begin,pfloat end);
    unsigned int getRecordsNum() const{return recordNum;}
    void printInfo();
};
class BayesClassifier{
protected:
    virtual float CalculateClassProbability() = 0;
public:
    BayesClassifier();
    ~BayesClassifier(){}
    virtual Classes Predict(const std::vector<cv::Mat>& channels) = 0;
    virtual void Train(const StaticPara* densityParas,float* classProbs) = 0;
};
class BasicNaiveBayesClassifier : public BayesClassifier{
    float T;
    std::vector<float> mu,sigma;
protected:
    float CalculateClassProbability();
public:
    BasicNaiveBayesClassifier();
    ~BasicNaiveBayesClassifier(){}
    Classes Predict(const std::vector<cv::Mat>& channels);
    void Train(const StaticPara* densityParas,float* classProbs);
};
}
#endif