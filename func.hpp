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
bool CalcChannelMeans(const std::vector<cv::Mat> & channels, std::map<Classes,float> & means);
};
namespace bayes{
class StaticPara{
    std::vector<float> mu,sigma;
    Classes classID;
    unsigned int recordNum;
public:
    StaticPara() = default;
    StaticPara(Classes classID):recordNum(0),classID(classID){}
    ~StaticPara(){}
    void InitClassType(Classes ID);
    void Sampling(const std::string& entryPath);
    float CombineMu(int begin,int end) const;
    float CombineSigma(int begin,int end) const;
    unsigned int getRecordsNum() const{return recordNum;}
    void printInfo();
    Classes getClassID() const{return classID;}
    const std::vector<float>& getMu() const{return mu;}
    const std::vector<float>& getSigma() const{return sigma;}
};
float CalcConv(const std::vector<float>& x, float avgx, const std::vector<float>& y, float avgy);
template <class paraForm>
class BayesClassifier{
protected:
    float T;
    std::map<Classes,paraForm> para;
    virtual float CalculateClassProbability(unsigned int classID,const std::map<Classes,float>& x) = 0;
public:
    BayesClassifier(){}
    ~BayesClassifier(){}
    virtual Classes Predict(const std::map<Classes,float>& x) = 0;
    virtual void Train(const StaticPara* densityParas,const float* classProbs) = 0;
};
struct BasicParaList{
    float mu,sigma,w;
};
class NaiveBayesClassifier : public BayesClassifier<BasicParaList>{
protected:
    float CalculateClassProbability(unsigned int classID,const std::map<Classes,float>& x);
    const static float lambda = 0.1f;//regularization parameter
public:
    NaiveBayesClassifier(){};
    ~NaiveBayesClassifier(){}
    Classes Predict(const std::map<Classes,float>& x);
    void Train(const StaticPara* densityParas,const float* classProbs);
};
class NonNaiveBayesClassifier : public NaiveBayesClassifier{
protected:
    float CalculateClassProbability(unsigned int classID,const std::map<Classes,float>& x);
    float** convMat;
    float** invConvMat;
public:
    NonNaiveBayesClassifier(){convMat = nullptr; invConvMat = nullptr;}
    ~NonNaiveBayesClassifier(){delete[] convMat; delete[] invConvMat;}
    Classes Predict(const std::map<Classes,float>& x);
    void Train(const StaticPara* densityParas,const float* classProbs);
};
}
#endif