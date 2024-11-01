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
float calcMean(const vFloat& data){
    double sum = 0.0f;
    for (vFloat::const_iterator it = data.begin(); it != data.end(); it++)
        sum += *it;
    return static_cast<float>(sum / data.size());
}
public:
    T_Sample(classType label,vFloat featureData):label(label),features(featureData){}
    ~T_Sample(){}
    classType getLabel() const{return label;}
    const vFloat& getFeatures() const{return features;}
};
template <typename classType>
class T_Classifier{
protected:
    size_t featureNum;
    std::string outputPhotoName;
public:
    virtual classType Predict(const vFloat& x) = 0;
    const std::string& printPhoto() const{return outputPhotoName;}
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
        classType bestClass = classType::Unknown;
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
    double CalculateClassProbability(unsigned int classID,const vFloat& x);
    size_t getClassNum() const;
public:
    T_NaiveBayesClassifier(){this->outputPhotoName = "naiveBayes.png";};
    ~T_NaiveBayesClassifier(){}
    void Train(const std::vector<T_Sample<classType>>& samples,const float* classProbs);
};
template <typename classType>
class T_NonNaiveBayesClassifier : public T_BayesClassifier<convParaList,classType>{
protected:
    size_t getClassNum() const;
    double CalculateClassProbability(unsigned int classID,const vFloat& x);
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
    void Train(const std::vector<T_Sample<classType>>& samples,const float* classProbs);
};
}
namespace linear{
template <class classType>
class T_FisherClassifier : public T_Classifier<classType>{
    fMat projMat;
    void CalcSwSb(float** Sw,float** Sb,const std::vector<T_Sample<classType>>& samples);
    size_t getClassNum() const;
public:
    T_FisherClassifier() {projMat = nullptr;this->outputPhotoName = "fisher.png";}
    ~T_FisherClassifier(){
        if (projMat != nullptr){
            for (size_t i = 0; i < getClassNum(); i++)
                delete[] projMat[i];
            delete[] projMat;
        }
    }
    void Train(const std::vector<T_Sample<classType>>& samples);
    classType Predict(const vFloat& x);
};
}
#endif