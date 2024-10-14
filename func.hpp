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
typedef std::vector<Classes> vClasses;
typedef std::vector<float> vFloat;

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
bool CalcChannelMeans(const std::vector<cv::Mat> & channels, vFloat & means);
};
namespace bayes{
class StaticPara{
    std::vector<vFloat> mu,sigma;
    Classes classID;
    unsigned int recordNum;
public:
    StaticPara() = default;
    StaticPara(Classes classID):recordNum(0),classID(classID){}
    ~StaticPara(){}
    void InitClassType(Classes ID);
    void Sampling(const std::string& entryPath);
    vFloat CombineMu(int begin,int end) const;
    vFloat CombineSigma(int begin,int end) const;
    unsigned int getRecordsNum() const{return recordNum;}
    void printInfo();
    Classes getClassID() const{return classID;}
    const std::vector<vFloat>& getMu() const{return mu;}
    const std::vector<vFloat>& getSigma() const{return sigma;}
};
vFloat CalcConv(const std::vector<vFloat>& x, vFloat avgx, const std::vector<vFloat>& y, vFloat avgy);
template <class paraForm>
class BayesClassifier{
protected:
    float T;
    std::map<Classes,paraForm> para;
    virtual float CalculateClassProbability(unsigned int classID,const vFloat& x) = 0;
public:
    BayesClassifier(){}
    ~BayesClassifier(){}
    virtual Classes Predict(const vFloat& x) = 0;
    virtual void Train(const StaticPara* densityParas,const float* classProbs) = 0;
};
struct BasicParaList{
    vFloat mu,sigma;
    float w;
};
class NaiveBayesClassifier : public BayesClassifier<BasicParaList>{
protected:
    float CalculateClassProbability(unsigned int classID,const vFloat& x);
    const static float lambda = 0.1f;//regularization parameter
public:
    NaiveBayesClassifier(){};
    ~NaiveBayesClassifier(){}
    Classes Predict(const vFloat& x);
    void Train(const StaticPara* densityParas,const float* classProbs);
};
class NonNaiveBayesClassifier : public NaiveBayesClassifier{
protected:
    float CalculateClassProbability(unsigned int classID,const vFloat& x);
    vFloat** convMat;
    vFloat** invConvMat;
public:
    NonNaiveBayesClassifier(){convMat = nullptr; invConvMat = nullptr;}
    ~NonNaiveBayesClassifier(){delete[] convMat; delete[] invConvMat;}
    Classes Predict(const vFloat& x);
    void Train(const StaticPara* densityParas,const float* classProbs);
};
}
#endif