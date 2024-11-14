#ifndef CLASSFIERHPP
#define CLASSFIERHPP
#include "../func.hpp"
#include "../t_classifier.hpp"

template <class classType>
class T_NonNaiveBayesClassifier : public T_BayesClassifier<ConvBayesParaList,classType>{
using Dataset = vector<T_Sample<classType>>;
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x) override{
        const unsigned int classNum = this->getClassNum();
        double res = log(this->para[static_cast<classType>(classID)].w);
        res -= log(determinant(this->para[classID].convMat))/2;
        vFloat pd = x;
        for (unsigned int d = 0; d < this->featureNum; d++)
            pd[d] = x[d] - this->para[static_cast<classType>(classID)].mu[d];
        vFloat sumX(this->featureNum, 0.0);
        for (size_t i = 0; i < this->featureNum; i++)
            for (size_t j = 0; j < this->featureNum; j++)
                sumX[i] += x[j] * this->para[classID].invMat[i][j];
        for (unsigned int d = 0; d < this->featureNum; d++){
            float sum = 0.0f;
            for (unsigned int j = 0; j < this->featureNum; j++)
                sum += sumX[j] * x[j];
            res -= sum / 2;
        }
        return res;
    }
    double CalcConv(const vFloat& x, const vFloat& y){
        double res = 0;
        const size_t n = x.size();
        double xAvg = 0.0f, yAvg = 0.0f;
        for (size_t i = 0; i < n; i++){
            xAvg += x[i];
            yAvg += y[i];
        }
        xAvg /= n;    yAvg /= n;
        for (size_t i = 0; i < n; i++)
            res += (x[i] - xAvg) * (y[i] - yAvg);
        res /= (n-1);
        return res;
    }
    void CalcConvMat(fMat convMat,fMat invMat,const vector<vFloat>& bucket){
        for (size_t i = 0; i < this->featureNum; i++){
            convMat[i][i] = CalcConv(bucket[i],bucket[i]) + lambda;
            for (size_t j = i+1; j < this->featureNum; j++){
                double conv = CalcConv(bucket[i],bucket[j]);
                convMat[i][j] = conv * (1.0f - lambda);
                convMat[j][i] = conv * (1.0f - lambda);
            }
        }
        tcb::CalcInvMat(convMat,invMat,this->featureNum);
        return;
    }
    void LUdecomposition(fMat matrix, fMat L, fMat U){
        for (int i = 0; i < this->featureNum; i++) { // init LU
            for (int j = 0; j < this->featureNum; j++) {
                L[i][j] = 0;
                U[i][j] = matrix[i][j];
            }
            L[i][i] = 1;
        }
        for (int i = 0; i < this->featureNum; i++) { // LU decomposition
            for (int j = i; j < this->featureNum; j++) 
                for (int k = 0; k < i; ++k) 
                    U[i][j] -= L[i][k] * U[k][j];
            for (int j = i + 1; j < this->featureNum; j++) {
                for (int k = 0; k < i; ++k)
                    L[j][i] -= L[j][k] * U[k][i];
                L[j][i] = U[j][i] / U[i][i];
            }
        }
    }
    double determinant(fMat matrix) {
        fMat L = new float*[this->featureNum];
        fMat U = new float*[this->featureNum];
        for (size_t i = 0; i < this->featureNum; i++){
            L[i] = new float[this->featureNum];
            U[i] = new float[this->featureNum];
            for (size_t j = 0; j < this->featureNum; j++)
                L[i][j] = U[i][j] = 0;
        }
        LUdecomposition(matrix, L, U);
        double det = 1.0;
        for (int i = 0; i < this->featureNum; i++)
            det *= U[i][i];
        for (size_t i = 0; i < this->featureNum; i++){
            delete[] L[i];
            delete[] U[i];
        }
        delete[] L;
        delete[] U;
        return det;
    }
    static constexpr float lambda = 0.0f;//regularization parameter
public:
    T_NonNaiveBayesClassifier(){this->classifierName = "nonNaiveBayes";}
    ~T_NonNaiveBayesClassifier(){
        for (vector<ConvBayesParaList>::const_iterator it = this->para.begin(); it != this->para.end(); it++){
            if(it->convMat != nullptr){
                for(size_t i = 0;i < this->featureNum;i++)
                    delete[] it->convMat[i];
                delete[] it->convMat;
            }
            if(it->invMat != nullptr){
                for(size_t i = 0;i < this->featureNum;i++)
                    delete[] it->invMat[i];
                delete[] it->invMat;
            }
        }
    }
};
namespace fruit{
constexpr int classifierKernelSize = 9;
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
bool GenerateFeatureChannels(const cv::Mat &image,std::vector<cv::Mat> &channels);
typedef std::vector<Classes> vClasses;
}
using namespace fruit;
typedef T_StaticPara<Classes> StaticPara;
typedef T_Sample<Classes> Sample;
class NaiveBayesClassifier : public T_NaiveBayesClassifier<Classes>{
    size_t getClassNum() const override{return Classes::counter;}
    bool CalcClassProb(float* prob) override;
//public:
    //void Train(const std::vector<Sample>& samples,const float* classProbs) override;
};
class NonNaiveBayesClassifier : public T_NonNaiveBayesClassifier<Classes>{
    size_t getClassNum() const override{return Classes::counter;}  
    bool CalcClassProb(float* prob) override;
public:
    void train(const std::vector<Sample>& samples,const float* classProbs) override;
};
class FisherClassifier : public T_FisherClassifier<Classes>{
    size_t getClassNum() const override{return Classes::counter;}
};
bool CalcChannelMeanStds(const vector<cv::Mat> & channels, vFloat & data);
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
            fruit::GenerateFeatureChannels(sample, channels);
            CalcChannelMeanStds(channels, data);
            rowClasses.push_back(classifer->Predict(data));
        }
        patchClasses.push_back(rowClasses);
    }
    return true;
}
bool LinearClassify(const cv::Mat& rawimage,FisherClassifier* classifer,std::vector<std::vector<Classes>>& patchClasses); 
#endif