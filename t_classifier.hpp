#ifndef TCLASSIFIER_HPP
#define TCLASSIFIER_HPP
#include <opencv2/opencv.hpp>
#include <cstring>
#include <string>
#include <map>
#include <Eigen/Dense>
#include "func.hpp"

constexpr int classifierKernelSize = 9;
template <typename classType>
class T_StaticPara{
    std::vector<vFloat> avg,var;
    classType classID;
    unsigned int recordNum;
public:
    T_StaticPara() = default;
    T_StaticPara(classType classID):recordNum(0),classID(classID){}
    ~T_StaticPara(){}
    void InitClassType(classType ID);
    void Sampling(const std::string& entryPath);
    unsigned int getRecordsNum() const{return recordNum;}
    classType getClassID() const{return classID;}
    const std::vector<vFloat>& getAvg() const{return avg;}
    const std::vector<vFloat>& getVar() const{return var;}
};
template <typename classType>
class T_Sample{
    classType label;
    vFloat features;
    bool isTrain;
    float calcMean(const vFloat& data){
        double sum = 0.0f;
        for (vFloat::const_iterator it = data.begin(); it != data.end(); it++)
            sum += *it;
        return static_cast<float>(sum / data.size());
    }
public:
    T_Sample(classType label,vFloat featureData, bool isTrainSample = true):label(label),features(featureData),isTrain(isTrainSample){}
    ~T_Sample(){}
    classType getLabel() const{return label;}
    const vFloat& getFeatures() const{return features;}
};
template <typename classType>
class T_Classifier{
protected:
    size_t featureNum;
    std::string outputPhotoName;
    float precision,recall,f1;
public:
    virtual classType Predict(const vFloat& x) = 0;
    virtual size_t getClassNum() const = 0;
    const std::string& printPhoto() const{return outputPhotoName;}
    void Examine(const std::vector<T_Sample<classType>>& samples){
        size_t pcorrectNum = 0, rcorrectNum = 0;
        for (std::vector<T_Sample<classType>>::const_iterator it = samples.begin(); it != samples.end(); it++){
            if (it->isTrainSample())
                continue;
            if (Predict(it->getFeatures()) == it->getLabel())
                pcorrectNum++;
            if (it->getLabel() == Predict(it->getFeatures()))
                rcorrectNum++;
        }
        precision = static_cast<float>(pcorrectNum)/samples.size();
        recall = static_cast<float>(rcorrectNum)/samples.size();
        f1 = 2*precision*recall/(precision+recall);
    }
    void PrintPrecision(){
        std::cout << "Precision: " << precision << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "F1: " << f1 << std::endl;
    }
};
namespace bayes{
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
template <class paraForm,typename classType>
class T_BayesClassifier : public T_Classifier<classType>{
protected:
    std::vector<paraForm> para;
    virtual double CalculateClassProbability(unsigned int classID,const vFloat& x) = 0;
public:
    T_BayesClassifier(){}
    ~T_BayesClassifier(){}
    classType Predict(const vFloat& x){
        double maxProb = -10e9;
        classType bestClass;
        for (unsigned int classID = 0; classID < classType::counter; classID++){
            double prob = CalculateClassProbability(classID,x);
            if (prob > maxProb) {
                maxProb = prob;
                bestClass = static_cast<classType>(classID);
            }
        }
        return bestClass;
    }
    virtual void Train(const std::vector<T_Sample<classType>>& samples,const float* classProbs) = 0;
};
template <typename classType>
class T_NaiveBayesClassifier : public T_BayesClassifier<BasicParaList,classType>{
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x){
        const unsigned int classNum = getClassNum();
        double res = log(this->para[static_cast<classType>(classID)].w);
        res -= log(determinant(this->para[classID].convMat))/2;
        vFloat pd = x;
        for (unsigned int d = 0; d < featureNum; d++)
            pd[d] = x[d] - this->para[static_cast<classType>(classID)].mu[d];
        vFloat sumX(featureNum, 0.0);
        for (size_t i = 0; i < featureNum; i++)
            for (size_t j = 0; j < featureNum; j++)
                sumX[i] += x[j] * this->para[classID].invMat[i][j];
        for (unsigned int d = 0; d < featureNum; d++){
            float sum = 0.0f;
            for (unsigned int j = 0; j < featureNum; j++)
                sum += sumX[j] * x[j];
            res -= sum / 2;
        }
        return res;
    }
public:
    T_NaiveBayesClassifier(){this->outputPhotoName = "naiveBayes.png";};
    ~T_NaiveBayesClassifier(){}
};
template <typename classType>
class T_NonNaiveBayesClassifier : public T_BayesClassifier<convParaList,classType>{
protected:
    virtual double CalculateClassProbability(unsigned int classID,const vFloat& x);
    void CalcConvMat(fMat convMat,fMat invMat,const std::vector<vFloat>& bucket){
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
    T_NonNaiveBayesClassifier(){this->outputPhotoName = "nonNaiveBayes.png";}
    ~T_NonNaiveBayesClassifier(){
        for (std::vector<convParaList>::const_iterator it = this->para.begin(); it != this->para.end(); it++){
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
}
namespace linear{
template <class classType>
class T_FisherClassifier : public T_Classifier<classType>{
protected:
    using SampleType = T_Sample<classType>;
    fMat projMat;
    void CalcSwSb(float** Sw,float** Sb,const std::vector<SampleType>& samples){
        unsigned int classNum = getClassNum();
        vFloat featureAvg(featureNum, 0.0f);
        std::vector<double> classifiedFeaturesAvg[classNum];
        for (int i = 0; i < classNum; i++)
            classifiedFeaturesAvg[i].assign(featureNum,0.0);
        std::vector<size_t> classRecordNum(classNum,0);
        for (std::vector<SampleType>::const_iterator it = samples.begin(); it != samples.end(); it++){
            unsigned int label = static_cast<unsigned int>(it->getLabel());
            const vFloat& sampleFeature = it->getFeatures();
            for (unsigned int i = 0; i < featureNum; i++)
                classifiedFeaturesAvg[label][i] += sampleFeature[i];
            classRecordNum[label]++;
        }
        for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < featureNum; j++){
            classifiedFeaturesAvg[i][j] /= classRecordNum[i];
            featureAvg[j] += classifiedFeaturesAvg[i][j];
        }
        for (unsigned int j = 0; j < featureNum; j++)
            featureAvg[j] /= classNum;

        for (std::vector<SampleType>::const_iterator it = samples.begin(); it != samples.end(); it++){
            unsigned int label = it->getLabel();
            const vFloat& sampleFeature = it->getFeatures();
            vFloat pd = sampleFeature;
            for (int j = 0; j < featureNum; j++) 
                pd[j] -= classifiedFeaturesAvg[label][j];
            for (int j = 0; j < featureNum; ++j)
                for (int k = 0; k < featureNum; ++k)
                    Sw[j][k] += pd[j] * pd[k];
        }
        for (int i = 0; i < classNum; i++) {
            vFloat pd(featureNum, 0.0f);
            for (int j = 0; j < featureNum; j++)
                pd[j] = classifiedFeaturesAvg[i][j] - featureAvg[j];
            for (int j = 0; j < featureNum; j++)
                for (int k = 0; k < featureNum; k++)
                    Sb[j][k] += classRecordNum[i] * pd[j] * pd[k];
        }
    }
public:
    T_FisherClassifier() {projMat = nullptr;this->outputPhotoName = "fisher.png";}
    ~T_FisherClassifier(){
        if (projMat != nullptr){
            for (size_t i = 0; i < getClassNum(); i++)
                delete[] projMat[i];
            delete[] projMat;
        }
    }
    virtual void Train(const std::vector<SampleType>& samples) = 0;
    classType Predict(const vFloat& x){
        classType resClass;
        double maxProb = -10e9;
        for (unsigned int classID = 0; classID < getClassNum(); classID++){
            double prob = 0.0f;
            for (unsigned int i = 0; i < featureNum; i++)
                prob += x[i] * projMat[classID][i];
            if (prob > maxProb){
                maxProb = prob;
                resClass = static_cast<classType>(classID);
            }
        }
        return resClass;
    }
};
}
template <class classType>
class T_SVMClassifier : public T_Classifier<classType>{
public:
    T_SVMClassifier() {this->outputPhotoName = "svm.png";}
    ~T_SVMClassifier(){}
    virtual void Train(const std::vector<T_Sample<classType>>& samples) = 0;
    classType Predict(const vFloat& x){
        classType resClass;
        return resClass;
    }
};
template <class classType>
class T_BPClassifier : public T_Classifier<classType>{
public:
    T_BPClassifier() {this->outputPhotoName = "bp.png";}
    ~T_BPClassifier(){}
    virtual void Train(const std::vector<T_Sample<classType>>& samples) = 0;
    classType Predict(const vFloat& x){
        classType resClass;
        return resClass;
    }
};
template <class classType>
class T_RandomForestClassifier : public T_Classifier<classType>{
public:
    T_RandomForestClassifier() {this->outputPhotoName = "rf.png";}
    ~T_RandomForestClassifier(){}
    virtual void Train(const std::vector<T_Sample<classType>>& samples) = 0;
    classType Predict(const vFloat& x){
        classType resClass;
        return resClass;
    }
};
#endif