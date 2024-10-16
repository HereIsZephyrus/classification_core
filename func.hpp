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
    hue,
    saturation,
    value,
    //gradient,
    angle,
    dim
};
constexpr int classifierKernelSize = 9;
typedef std::vector<Classes> vClasses;
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
};
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
class Classifier{
protected:
    size_t featureNum;
    std::string outputPhotoName;
public:
    virtual Classes Predict(const vFloat& x) = 0;
    const std::string& printPhoto() const{return outputPhotoName;}
};
namespace bayes{
double CalcConv(const vFloat& x, const vFloat& y);
template <class paraForm>
class BayesClassifier : public Classifier{
protected:
    std::vector<paraForm> para;
    virtual double CalculateClassProbability(unsigned int classID,const vFloat& x) = 0;
public:
    BayesClassifier(){}
    ~BayesClassifier(){}
    Classes Predict(const vFloat& x){
        double maxProb = -10e9;
        Classes bestClass = Classes::Unknown;
        for (unsigned int classID = 0; classID < Classes::counter; classID++){
            double prob = CalculateClassProbability(classID,x);
            if (prob > maxProb) {
                maxProb = prob;
                bestClass = static_cast<Classes>(classID);
            }
        }
        return bestClass;
    }
    virtual void Train(const std::vector<Sample>& samples,const float* classProbs) = 0;
};
struct BasicParaList{
    float w;
    std::vector<double> mu,sigma;
};
struct convParaList{
    float w;
    std::vector<double> mu;
    fMat convMat;
    fMat invMat;
};
class NaiveBayesClassifier : public BayesClassifier<BasicParaList>{
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x);
public:
    NaiveBayesClassifier(){outputPhotoName = "naiveBayes.png";};
    ~NaiveBayesClassifier(){}
    void Train(const std::vector<Sample>& samples,const float* classProbs);
};
class NonNaiveBayesClassifier : public BayesClassifier<convParaList>{
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x);
    void CalcConvMat(fMat convMat,fMat invMat,const std::vector<vFloat>& bucket);
    void LUdecomposition(fMat matrix, fMat L, fMat U);
    double determinant(fMat matrix);
    static constexpr float lambda = 0.0f;//regularization parameter
public:
    NonNaiveBayesClassifier(){outputPhotoName = "nonNaiveBayes.png";}
    ~NonNaiveBayesClassifier();
    void Train(const std::vector<Sample>& samples,const float* classProbs);
};
}
namespace linear{
class FisherClassifier : public Classifier{
    fMat projMat;
    void CalcSwSb(float** Sw,float** Sb,const std::vector<Sample>& samples);
public:
    FisherClassifier() {projMat = nullptr;outputPhotoName = "fisher.png";}
    ~FisherClassifier();
    void Train(const std::vector<Sample>& samples);
    Classes Predict(const vFloat& x);
};
bool LinearClassify(const cv::Mat& rawimage,FisherClassifier* classifer,std::vector<std::vector<Classes>>& patchClasses);
};
#endif