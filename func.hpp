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
extern std::unordered_map<Classes,cv::Scalar> classifyColor;
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
bool CalcChannelMeanStds(const std::vector<cv::Mat> & channels, vFloat & data);
};
namespace bayes{
class StaticPara{
    std::vector<vFloat> avg,var;
    Classes classID;
    unsigned int recordNum;
public:
    StaticPara() = default;
    StaticPara(Classes classID):recordNum(0),classID(classID){}
    ~StaticPara(){}
    void InitClassType(Classes ID);
    void Sampling(const std::string& entryPath);
    unsigned int getRecordsNum() const{return recordNum;}
    Classes getClassID() const{return classID;}
    const std::vector<vFloat>& getAvg() const{return avg;}
    const std::vector<vFloat>& getVar() const{return var;}
};
class Sample{
Classes label;
vFloat features;
float calcMean(const vFloat& data);
public:
    Sample(Classes label,vFloat featureData):label(label),features(featureData){}
    ~Sample(){}
    Classes getLabel() const{return label;}
    const vFloat& getFeatures() const{return features;}
};
vFloat CalcConv(const std::vector<vFloat>& x, vFloat avgx, const std::vector<vFloat>& y, vFloat avgy);
template <class paraForm>
class BayesClassifier{
protected:
    float T;
    size_t featureNum;
    std::vector<paraForm> para;
    virtual double CalculateClassProbability(unsigned int classID,const vFloat& x) = 0;
public:
    BayesClassifier(){}
    ~BayesClassifier(){}
    virtual Classes Predict(const vFloat& x) = 0;
    virtual void Train(const std::vector<Sample>& samples,const float* classProbs) = 0;
    void setFeatureNum(size_t num){featureNum = num;}
};
struct BasicParaList{
    float w;
    std::vector<double> mu,sigma;
};
class NaiveBayesClassifier : public BayesClassifier<BasicParaList>{
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x);
public:
    NaiveBayesClassifier(){};
    ~NaiveBayesClassifier(){}
    Classes Predict(const vFloat& x);
    void Train(const std::vector<Sample>& samples,const float* classProbs);
};
/*
class NonNaiveBayesClassifier : public NaiveBayesClassifier{
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x);
    static constexpr float lambda = 0.1f;//regularization parameter
    vFloat** convMat;
    vFloat** invConvMat;
public:
    NonNaiveBayesClassifier(){convMat = nullptr; invConvMat = nullptr;}
    ~NonNaiveBayesClassifier(){delete[] convMat; delete[] invConvMat;}
    Classes Predict(const vFloat& x);
    void Train(const StaticPara* densityParas,const float* classProbs);
};
*/
}
#endif