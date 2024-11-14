#ifndef TCLASSIFIER_HPP
#define TCLASSIFIER_HPP
#define CALC_EDGE false
#include <cstring>
#include <string>
#include <cstdlib>
#include <ctime>
#include <map>
#include <memory>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "func.hpp"

constexpr int defaultClassifierKernelSize = 9;
using std::vector;
using std::pair;
using std::map;
using namespace Eigen;
template <class classType>
class T_StaticPara{
    vector<vFloat> avg,var;
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
    const vector<vFloat>& getAvg() const{return avg;}
    const vector<vFloat>& getVar() const{return var;}
};
template <class classType>
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
    void scalingFeatures(const vFloat& maxVal,const vFloat& minVal){
        for (size_t i = 0; i < features.size(); i++)
            features[i] = (features[i] - minVal[i])  / (maxVal[i] - minVal[i]);
    }
    bool isTrainSample() const{return isTrain;}
};
template <class classType>
class T_Classifier{
using Dataset = vector<T_Sample<classType>>;
protected:
    size_t featureNum;
    std::string outputPhotoName;
    float precision,recall,f1;
public:
    virtual classType Predict(const vFloat& x) = 0;
    virtual size_t getClassNum() const{return 0;}
    virtual void Train(const Dataset &dataset) = 0;
    const std::string& printPhoto() const{return outputPhotoName;}
    void Examine(const Dataset& samples){
        size_t TP = 0, FP = 0,FN = 0,testSampleNum = 0;
        for (typename Dataset::const_iterator it = samples.begin(); it != samples.end(); it++){
            if (it->isTrainSample())
                continue;
            ++ testSampleNum;
            if (Predict(it->getFeatures()) == it->getLabel())
                TP++;
            if (it->getLabel() == Predict(it->getFeatures()))
                FP++;
        }
        precision = static_cast<float>(TP)/testSampleNum;
        recall = static_cast<float>(FP)/testSampleNum;
        f1 = 2*precision*recall/(precision+recall);
    }
    void PrintPrecision(){
        std::cout << "Precision: " << precision << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "F1: " << f1 << std::endl;
    }
    void Classify(const cv::Mat& featureImage,vector<vector<classType>>& pixelClasses,classType edgeType,const vFloat& minVal,const vFloat& maxVal,int classifierKernelSize = defaultClassifierKernelSize){
        int classNum = getClassNum();
        using ClassMat = vector<vector<classType>>;
        using vClasses = vector<classType>;
        int rows = featureImage.rows, cols = featureImage.cols;
        vector<vector<classType>> patchClasses;
        for (int r = classifierKernelSize/2; r <= rows - classifierKernelSize; r+=classifierKernelSize/2){
            vClasses rowClasses;
            bool lastRowCheck = (r >= (rows - classifierKernelSize));
            for (int c = classifierKernelSize/2; c <= cols - classifierKernelSize; c+=classifierKernelSize/2){
                bool lastColCheck = (c >= (cols - classifierKernelSize));
                cv::Rect window(c - classifierKernelSize/2 ,r - classifierKernelSize/2,  classifierKernelSize - lastColCheck, classifierKernelSize - lastRowCheck);  
                cv::Mat sample = featureImage(window);
                vector<cv::Mat> channels;
                cv::split(sample, channels);
                vFloat data;
                tcb::CalcChannelMeanStds(channels, data);
                for (size_t i = 0; i < data.size(); i++)
                    data[i] = (data[i] - minVal[i])  / (maxVal[i] - minVal[i]);
                rowClasses.push_back(Predict(data));
            }
            patchClasses.push_back(rowClasses);
        }
        typename ClassMat::const_iterator row = patchClasses.begin();
        { //tackle the first line
            vClasses temprow;
            for (typename vClasses::const_iterator col = row->begin(); col != row->end(); col++){
                if (col == row->begin()){
                    temprow.push_back(*col);
                    continue;
                }
                if (CALC_EDGE &&(*col) != (*(col-1)))//horizontalEdgeCheck
                    temprow.push_back(edgeType);
                else
                    temprow.push_back(*col);
                if ((col+1) == row->end())// manually add the last element
                    temprow.push_back(*col);
            }
            pixelClasses.push_back(temprow);
        }
        typename vClasses::const_iterator lastRowBegin = row->begin();
        row++;
        for (; row != patchClasses.end(); row++){
            vClasses temprow;
            typename vClasses::const_iterator col = row->begin(),diagonalCol = lastRowBegin;
            { //tackle the first col
                if (!CALC_EDGE || *col == *diagonalCol)
                    temprow.push_back(*col);
                else
                    temprow.push_back(edgeType);
                col++;
            }
            for (;col != row->end(); col++,diagonalCol++){
                bool horizontalEdgeCheck = (*col) == (*(col-1));
                bool verticalEdgeCheck = (*col) == (*(diagonalCol+1));
                bool diagonalEdgeCheck = (*col) == (*(diagonalCol));
                if (!CALC_EDGE || (horizontalEdgeCheck && verticalEdgeCheck && diagonalEdgeCheck))
                    temprow.push_back(*col);
                else
                    temprow.push_back(edgeType);
                if ((col+1) == row->end())// manually add the last element
                    temprow.push_back(*col);
            }
            pixelClasses.push_back(temprow);
            if (row+1 != patchClasses.end())//pause on the last row
                lastRowBegin = row->begin();
        }
        row--;
        {// manually add the last row
            vClasses temprow;
            typename vClasses::const_iterator col = row->begin(),diagonalCol = lastRowBegin;
            { //tackle the first col
                if (!CALC_EDGE || *col == *diagonalCol)
                    temprow.push_back(*col);
                else
                    temprow.push_back(edgeType);
                col++;
            }
            for (;col != row->end(); col++,diagonalCol++){
                bool horizontalEdgeCheck = (*col) == (*(col-1));
                bool verticalEdgeCheck = (*col) == (*(diagonalCol+1));
                bool diagonalEdgeCheck = (*col) == (*(diagonalCol));
                if (!CALC_EDGE || (horizontalEdgeCheck && verticalEdgeCheck && diagonalEdgeCheck))
                    temprow.push_back(*col);
                else
                    temprow.push_back(edgeType);
                if ((col+1) == row->end())// manually add the last element
                    temprow.push_back(*col);
            }
            pixelClasses.push_back(temprow);
        }
    }
};
struct BasicBayesParaList{
    float w;
    vector<double> mu,sigma;
};
struct ConvBayesParaList{
    float w;
    vector<double> mu;
    fMat convMat;
    fMat invMat;
};
template <class paraForm,class classType>
class T_BayesClassifier : public T_Classifier<classType>{
using Dataset = vector<T_Sample<classType>>;
protected:
    vector<paraForm> para;
    virtual double CalculateClassProbability(unsigned int classID,const vFloat& x) = 0;
    virtual bool CalcClassProb(float* prob) = 0;
public:
    T_BayesClassifier(){}
    ~T_BayesClassifier(){}
    classType Predict(const vFloat& x) override{
        unsigned int classNum = this->getClassNum();
        double maxProb = -10e9;
        classType bestClass;
        for (unsigned int classID = 0; classID < classNum; classID++){
            double prob = CalculateClassProbability(classID,x);
            if (prob > maxProb) {
                maxProb = prob;
                bestClass = static_cast<classType>(classID);
            }
        }
        return bestClass;
    }
    virtual void train(const Dataset& samples,const float* classProbs) = 0;
    virtual void Train(const Dataset& dataset) override{
        unsigned int classesNum = this->getClassNum();
        float* classProbs = new float[classesNum];
        CalcClassProb(classProbs);
        train(dataset,classProbs);
        delete[] classProbs;
    }
};
template <class classType>
class T_NaiveBayesClassifier : public T_BayesClassifier<BasicBayesParaList,classType>{
using Dataset = vector<T_Sample<classType>>;
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x) override{
        double res = this->para[static_cast<classType>(classID)].w;
        for (unsigned int d = 0; d < this->featureNum; d++){
            float pd = x[d] - this->para[static_cast<classType>(classID)].mu[d];
            float vars = this->para[static_cast<classType>(classID)].sigma[d] * this->para[static_cast<classType>(classID)].sigma[d];
            double exponent = exp(static_cast<double>(- pd * pd / (2 * vars)));
            double normalize = 1.0f / (sqrt(2 * CV_PI) * vars);
            res *= normalize * exponent;
        }
        return res;
    }
public:
    T_NaiveBayesClassifier(){this->outputPhotoName = "naiveBayes.png";};
    ~T_NaiveBayesClassifier(){}
    void train(const Dataset& dataset,const float* classProbs) override{
        this->featureNum = dataset[0].getFeatures().size();
        this->para.clear();
        unsigned int classNum = this->getClassNum();
        vector<double> classifiedFeaturesAvg[classNum],classifiedFeaturesVar[classNum];
        for (int i = 0; i < classNum; i++){
            classifiedFeaturesAvg[i].assign(this->featureNum,0.0);
            classifiedFeaturesVar[i].assign(this->featureNum,0.0);
        }
        vector<size_t> classRecordNum(classNum,0);
        for (typename Dataset::const_iterator it = dataset.begin(); it != dataset.end(); it++){
            if (!it->isTrainSample())
                continue;
            unsigned int label = static_cast<unsigned int>(it->getLabel());
            const vFloat& sampleFeature = it->getFeatures();
            for (unsigned int i = 0; i < this->featureNum; i++)
                classifiedFeaturesAvg[label][i] += sampleFeature[i];
            classRecordNum[label]++;
        }
        for (unsigned int i = 0; i < classNum; i++)
            for (unsigned int j = 0; j < this->featureNum; j++)
                classifiedFeaturesAvg[i][j] /= classRecordNum[i];
        for (typename Dataset::const_iterator it = dataset.begin(); it != dataset.end(); it++){
            unsigned int label = static_cast<unsigned int>(it->getLabel());
            const vFloat& sampleFeature = it->getFeatures();
            for (unsigned int i = 0; i < this->featureNum; i++)
                classifiedFeaturesVar[label][i] += (sampleFeature[i] - classifiedFeaturesAvg[label][i]) * (sampleFeature[i] - classifiedFeaturesAvg[label][i]);
        }
        for (unsigned int i = 0; i < classNum; i++)
            for (unsigned int j = 0; j < this->featureNum; j++)
                classifiedFeaturesVar[i][j] = std::sqrt(classifiedFeaturesVar[i][j]/classRecordNum[i]);
        for (unsigned int i = 0; i < classNum; i++){
            BasicBayesParaList temp;
            temp.w = classProbs[i];
            temp.mu = classifiedFeaturesAvg[i];
            temp.sigma = classifiedFeaturesVar[i];
            this->para.push_back(temp);
            /*
            {
                std::cout<<"class "<<i<<" counts"<<classRecordNum[i]<<std::endl;
                std::cout<<"w: "<<temp.w<<std::endl;
                std::cout<<"mu: ";
                for (unsigned int j = 0; j < this->featureNum; j++)
                    std::cout<<temp.mu[j]<<" ";
                std::cout<<std::endl;
                std::cout<<"sigma: ";
                for (unsigned int j = 0; j < this->featureNum; j++)
                    std::cout<<temp.sigma[j]<<" ";
                std::cout<<std::endl;
            }
            */
        }
        return;
    }
};
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
    T_NonNaiveBayesClassifier(){this->outputPhotoName = "nonNaiveBayes.png";}
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
template <class classType>
class T_FisherClassifier : public T_Classifier<classType>{
using Dataset = vector<T_Sample<classType>>;
protected:
    using SampleType = T_Sample<classType>;
    vector<vFloat> mu;
    vFloat signal,projMat;
    void CalcSwSb(float** Sw,float** Sb,const vector<SampleType>& samples){
        unsigned int classNum = this->getClassNum();
        vFloat featureAvg(this->featureNum, 0.0f);
        vFloat classifiedFeaturesAvg[classNum];
        for (int i = 0; i < classNum; i++)
            classifiedFeaturesAvg[i].assign(this->featureNum,0.0);
        vector<size_t> classRecordNum(classNum,0);
        int total = 0;
        for (typename Dataset::const_iterator it = samples.begin(); it != samples.end(); it++){
            if (!it->isTrainSample())
                continue;
            unsigned int label = static_cast<unsigned int>(it->getLabel());
            const vFloat& sampleFeature = it->getFeatures();
            for (unsigned int i = 0; i < this->featureNum; i++)
                classifiedFeaturesAvg[label][i] += sampleFeature[i];
            ++classRecordNum[label];
            ++total;
        }
        for (unsigned int i = 0; i < classNum; i++)
            for (unsigned int j = 0; j < this->featureNum; j++){
                featureAvg[j] += classifiedFeaturesAvg[i][j];
                classifiedFeaturesAvg[i][j] /= classRecordNum[i];
            }
        for (unsigned int j = 0; j < this->featureNum; j++)
            featureAvg[j] /= total;
        for (unsigned int i = 0; i < classNum; i++)
            mu.push_back(classifiedFeaturesAvg[i]);
        for (typename Dataset::const_iterator it = samples.begin(); it != samples.end(); it++){
            if (!it->isTrainSample())
                continue;
            unsigned int label = it->getLabel();
            const vFloat& sampleFeature = it->getFeatures();
            vFloat pd = sampleFeature;
            for (int j = 0; j < this->featureNum; j++) 
                pd[j] -= classifiedFeaturesAvg[label][j];
            for (int j = 0; j < this->featureNum; ++j)
                for (int k = 0; k < this->featureNum; ++k)
                    Sw[j][k] += pd[j] * pd[k];
        }
        for (int i = 0; i < classNum; i++) {
            vFloat pd(this->featureNum, 0.0f);
            for (int j = 0; j < this->featureNum; j++)
                pd[j] = classifiedFeaturesAvg[i][j] - featureAvg[j];
            for (int j = 0; j < this->featureNum; j++)
                for (int k = 0; k < this->featureNum; k++)
                    Sb[j][k] += classRecordNum[i] * pd[j] * pd[k];
        }
    }
public:
    T_FisherClassifier() {this->outputPhotoName = "fisher.png";}
    ~T_FisherClassifier(){}
    virtual void Train(const Dataset& dataset) override{
        this->featureNum = dataset[0].getFeatures().size();
        fMat SwMat,SbMat;
        SwMat = new float*[this->featureNum];
        SbMat = new float*[this->featureNum];
        for (size_t i = 0; i < this->featureNum; i++){
            SwMat[i] = new float[this->featureNum];
            SbMat[i] = new float[this->featureNum];
            for (size_t j = 0; j < this->featureNum; j++)
                SwMat[i][j] = 0.0f,SbMat[i][j] = 0.0f;
        }
        CalcSwSb(SwMat,SbMat,dataset);
        MatrixXf Sw(this->featureNum,this->featureNum),Sb(this->featureNum,this->featureNum);
        for (size_t i = 0; i < this->featureNum; i++)
            for (size_t j = 0; j < this->featureNum; j++)
                Sw(i,j) = SwMat[i][j],Sb(i,j) = SbMat[i][j];
        for (size_t i = 0; i < this->featureNum; i++){
            delete[] SwMat[i];
            delete[] SbMat[i];
        }
        delete[] SwMat;
        delete[] SbMat;
        SelfAdjointEigenSolver<MatrixXf> eig(Sw.completeOrthogonalDecomposition().pseudoInverse() * Sb);
        int classNum = this->getClassNum();
        MatrixXf projectionMatrix = eig.eigenvectors().rightCols(1);
        for (size_t j = 0; j < this->featureNum; j++){
            projMat.push_back(projectionMatrix(j));
        }
        for (int i = 0; i < classNum; i++){
            float calcmean = 0;
            for (size_t j = 0; j < this->featureNum; j++)
                calcmean += projectionMatrix(j) * mu[i][j];
            signal.push_back(calcmean);
            //std::cout<<signal[i]<<std::endl;
        }
        return;
    }
    classType Predict(const vFloat& x) override{
        classType resClass;
        double projed = 0;
        for (unsigned int i = 0; i < this->featureNum; i++)
            projed += x[i] * projMat[i];
        double minDistance = 1e10;
        for (unsigned int classID = 0; classID < this->getClassNum(); classID++){
            double distance = (projed - signal[classID])*(projed - signal[classID]);
            if (distance < minDistance) {
                minDistance = distance;
                resClass = static_cast<classType>(classID);
            }
        }
        return resClass;
    }
};
template <class classType>
class T_SVMClassifier : public T_Classifier<classType>{
using Dataset = vector<T_Sample<classType>>;
protected:
    class OVOSVM {
        double learningRate,bias,limit;
        int maxIter;
        vector<vFloat> supportVectors;
        vector<int> supportLabels;
        vector<double> supportAlpha;
        vFloat weights;
        classType positiveClass,negetiveClass;
        static constexpr double eps = 1e-6;
        double dot(const vFloat& x,const vFloat& y) {
            double result = 0.0;
            for (size_t i = 0; i < x.size(); i++)
                result += x[i] * y[i];
            return result;
        }
    public:
        OVOSVM(classType pos,classType neg,double limit = 0.01,double learningRate = 0.001, int maxIter = 1000)
        : learningRate(learningRate),maxIter(maxIter),positiveClass(pos),negetiveClass(neg) {}
        void train(const Dataset& dataset,const vector<int>& index, const vector<int>& y) {
            int sampleNum = index.size(),featureNum = dataset[index[0]].getFeatures().size();
            weights.assign(featureNum, 0.0);
            double beta = 1.0;
            bool notMargined = true;
            vFloat alpha;
            alpha.assign(sampleNum, 0.0);
            for (int iter = 0; iter < maxIter; ++iter) {
                notMargined = false;
                double error = 0;
                for (int i = 0; i < sampleNum; i++) {
                    double item1 = 0.0;
                    for (int j = 0; j < sampleNum; j++)
                        item1 += alpha[j] * (double)y[i] * (double)y[j] * dot(dataset[index[i]].getFeatures(), dataset[index[j]].getFeatures());
                    double item2 = 0.0;
                    for (int j = 0; j < sampleNum; j++)
                        item2 += alpha[j] * (double)y[i] * (double)y[j];
                    double delta = 1.0 - item1 - beta * item2;
                    alpha[i] += learningRate * delta;
                    alpha[i] = std::max(alpha[i],0.0f);
                    if (std::abs(delta) > limit){
                        notMargined = true;
                        error += std::abs(delta) - limit;
                    }
                }
                double item3 = 0.0;
                for (int i = 0; i < sampleNum; i++)
                    item3 += alpha[i] * (double)y[i];
                beta += item3 * item3 / 2.0;
                if (!notMargined)   break;
            }
            for (int i = 0; i < sampleNum; i++){
                if (alpha[i] > eps){
                    supportVectors.push_back(dataset[index[i]].getFeatures());
                    supportLabels.push_back(y[i]);
                    supportAlpha.push_back(alpha[i]);
                }
            }
            weights.assign(featureNum,0.0);
            for (int j = 0; j < featureNum; j++)
                for (size_t i = 0; i < supportVectors.size(); i++)
                    weights[j] += supportAlpha[i] * supportLabels[i] * supportVectors[i][j];
            bias = 0.0;
            for (size_t i = 0; i < supportVectors.size(); i++)
                bias += supportLabels[i] - dot(weights, supportVectors[i]);
            bias /= static_cast<double>(supportVectors.size());
        }
        bool predict(const vFloat& sample){return (dot(sample,weights) + bias)>0;}
        classType getPositiveClass() const{ return positiveClass; }
        classType getNegetiveClass() const{ return negetiveClass; }
    };
    vector<std::unique_ptr<OVOSVM>> classifiers;
public:
    T_SVMClassifier(){this->outputPhotoName = "svm.png";}
    ~T_SVMClassifier(){}
    virtual void Train(const Dataset& dataset) override{
        this->featureNum = dataset[0].getFeatures().size();
        unsigned int classNum = this->getClassNum();
        std::vector<int> classCount[classNum];
        for (size_t i = 0; i < dataset.size(); i++){
            const T_Sample<classType>& sample = dataset[i];
            if (!sample.isTrainSample())
                continue;
            unsigned int classID = static_cast<unsigned int>(sample.getLabel());
            classCount[classID].push_back(i);
        }
        for (unsigned int i = 0; i < classNum; i++)
            for (unsigned int j = i+1; j < classNum; j++){
                std::vector<int> classPN = classCount[i];
                std::vector<int> classLabeli,classLabelj;
                classLabeli.assign(classCount[i].size(),1);
                classLabelj.assign(classCount[j].size(),-1);
                classLabeli.insert(classLabeli.end(), classLabelj.begin(), classLabelj.end());
                classPN.insert(classPN.end(), classCount[j].begin(), classCount[j].end());
                std::unique_ptr<OVOSVM> classifier = std::make_unique<OVOSVM>(static_cast<classType>(i),static_cast<classType>(j));
                classifier->train(dataset,classPN,classLabeli);
                classifiers.push_back(std::move(classifier));
            }
    };
    classType Predict(const vFloat& x) override{
        vector<unsigned int> classVote(this->getClassNum());
        classVote.assign(this->getClassNum(),0);
        for (typename vector<std::unique_ptr<OVOSVM>>::iterator it = classifiers.begin(); it != classifiers.end(); it++){
            if ((*it)->getNegetiveClass() > this->getClassNum()){
                if ((*it)->predict(x))
                    return (*it)->getPositiveClass();
            }else{
                if ((*it)->predict(x))
                    ++classVote[static_cast<unsigned int>((*it)->getPositiveClass())];
                else
                    ++classVote[static_cast<unsigned int>((*it)->getNegetiveClass())];
            }
        }
        int maxVote = 0;
        classType resClass;
        for (vector<unsigned int>::const_iterator it = classVote.begin(); it != classVote.end(); it++)
            if (*it > maxVote){
                maxVote = *it;
                resClass = static_cast<classType>(it - classVote.begin());
            }
        return resClass;
    }
};
template <class classType>
class T_BPClassifier : public T_Classifier<classType>{
using Dataset = vector<T_Sample<classType>>;
protected:
    int classNum,hiddenSize;
    vector<vFloat> weightsInput2Hidden,weightsHidden2Output,deltaWeightsInput2Hidden,deltaWeightsHidden2Output;
    double learningRate,momentum;
    float activation(float x) {return 1.0 / (1.0 + exp(-x));}
    void initWeights(){
        weightsInput2Hidden.assign(this->featureNum+1,vFloat(hiddenSize,0));
        weightsHidden2Output.assign(hiddenSize+1,vFloat(classNum,0));
        deltaWeightsInput2Hidden.assign(this->featureNum+1,vFloat(hiddenSize,0));
        deltaWeightsHidden2Output.assign(hiddenSize+1,vFloat(classNum,0));
        double rangeHidden = 1/sqrt((double)this->featureNum);
        double rangeOutput = 1/sqrt((double)hiddenSize);
        srand(time(0));
        for (int i = 0; i <= this->featureNum; i++)
            for (int j = 0; j < hiddenSize; j++)
                weightsInput2Hidden[i][j] = (((double)(rand() % 100 + 1))/100.0) * 2.0 * rangeHidden - rangeHidden;
        for (int j = 0; j <= hiddenSize; j++)
            for (int k = 0; k < classNum; k++)
                weightsHidden2Output[j][k] = (((double)(rand() % 100 + 1))/100.0) * 2.0 * rangeOutput - rangeOutput;
    }
    void forwardFeed(const vFloat& inputs,vFloat& neuronHidden,vFloat& neuronOutput) {
        vFloat neuronInput = inputs;
        neuronInput.push_back(-1); // bias neuronn
        neuronHidden.assign(hiddenSize,0);
        neuronHidden.push_back(-1); // bias neuronn
        neuronOutput.assign(classNum,0);
        for (int j = 0; j < hiddenSize; j++){
            for (int i = 0; i <= this->featureNum; i++)
                neuronHidden[j] += neuronInput[i] * weightsInput2Hidden[i][j];
            neuronHidden[j] = activation(neuronHidden[j]);
        }
        for (int k = 0; k < classNum; k++){
            for (int j = 0; j <= hiddenSize; j++)
                neuronOutput[k] += neuronHidden[j] * weightsHidden2Output[j][k];
            neuronOutput[k] = activation(neuronOutput[k]);
        }
    }
    void backwardFeed(unsigned int loc,const vFloat& neuronInput,const vFloat& neuronHidden,const vFloat& neuronOutput) {
        vFloat idealOutput(classNum,0.0f);
        vFloat errorOutput(classNum,0.0f);
        vFloat errorHidden(hiddenSize+1,0.0f);
        idealOutput[loc] = 1.0f;
        for (int k = 0; k < classNum; k++)
            errorOutput[k] = neuronOutput[k] * (1 - neuronOutput[k]) * (idealOutput[k] - neuronOutput[k]);
        for (int j = 0; j <= hiddenSize; j++){
            int sum = 0;
            for (int k = 0; k < classNum; k++){
                sum += weightsHidden2Output[j][k] * errorOutput[k];
                deltaWeightsHidden2Output[j][k] = learningRate * neuronHidden[j] * errorOutput[k] + momentum * deltaWeightsHidden2Output[j][k];
                weightsHidden2Output[j][k] += deltaWeightsHidden2Output[j][k];
            }
            errorHidden[j] = neuronHidden[j] * (1 - neuronHidden[j]) * sum;
        }
        for (int i = 0; i <= this->featureNum; i++)
            for (int j = 0; j < hiddenSize; j++){
                deltaWeightsInput2Hidden[i][j] = learningRate * neuronInput[i] * errorHidden[j] + momentum * deltaWeightsInput2Hidden[i][j];
                weightsInput2Hidden[i][j] += deltaWeightsInput2Hidden[i][j];
            }
    }
public:
    T_BPClassifier(int hiddensize = 20, double rate = 0.4, double mom = 0.8):classNum(0),hiddenSize(hiddensize),learningRate(rate),momentum(mom) {this->outputPhotoName = "bp.png";}
    ~T_BPClassifier(){}
    virtual void Train(const Dataset& dataset) override{
        this->featureNum = dataset[0].getFeatures().size();
        classNum = this->getClassNum();
        initWeights();
        int maxiter = 100;
        while(maxiter--){
            double TMSE = 0,Tacc = 0;
            int total = 0;
            for (typename Dataset::const_iterator data = dataset.begin(); data != dataset.end(); data++){
                if (!data->isTrainSample())
                    continue;
                ++total;
                classType label = data->getLabel(),resClass;
                unsigned int uLabel = static_cast<unsigned int>(label);
                double maxVal = -1.0;
                vFloat hidden,output;
                forwardFeed(data->getFeatures(),hidden,output);
                TMSE += (1.0 - output[uLabel]) * (1.0 - output[uLabel]);
                for (int k = 0; k < classNum; k++)
                    if (output[k] > maxVal){
                        maxVal = output[k];
                        resClass = static_cast<classType>(k);
                    }
                if (label == resClass)  Tacc += 1.0;
                backwardFeed(uLabel,data->getFeatures(),hidden,output);
            }
            TMSE = TMSE / (double)total;
            Tacc = Tacc / (double)total * 100.0;
            if (TMSE < 0.05 || Tacc > 95)
                break;
        }
    }
    classType Predict(const vFloat& x) override{
        vFloat hidden,actived;
        forwardFeed(x,hidden,actived);
        classType resClass;
        float maxLight = 0.0f;
        for (int i = 0; i < classNum; i++)
            if (actived[i] > maxLight){
                maxLight = actived[i];
                resClass = static_cast<classType>(i);
            }
        return resClass;
    }
};
template <class classType>
class T_RandomForestClassifier : public T_Classifier<classType>{
using Dataset = vector<T_Sample<classType>>;
using ProbClass = map<classType,double>;
protected:
    class DecisionTree {
    private:
        struct Node {
            int featureIndex;
            std::shared_ptr<Node> left;
            std::shared_ptr<Node> right;
            float threshold;
            ProbClass prob;
            bool isLeaf;
            Node(bool isLeaf,int featureIndex = -1, float threshold = 0,std::shared_ptr<Node> l = nullptr,std::shared_ptr<Node> r = nullptr):
                isLeaf(isLeaf),left(l),right(r),featureIndex(featureIndex),threshold(threshold) {}
        };
        std::shared_ptr<Node> root;
        int featureNum;
        int maxDepth;
        int minSamplesSplit,minSamplesLeaf;
        double computeGiniIndex(const Dataset& dataset,vector<pair<int, double>> samplesFeaturesVec,size_t splitIndex) {
            map<classType,int> leftCounter,rightCounter;
            size_t totalSize = samplesFeaturesVec.size();
            for (size_t index = 0; index < splitIndex; index++)
                leftCounter[dataset[samplesFeaturesVec[index].first].getLabel()]++;
            for (size_t index = splitIndex; index < totalSize; index++)
                rightCounter[dataset[samplesFeaturesVec[index].first].getLabel()]++;
            double leftGini = 0.0,rightGini = 0.0;
            for (typename map<classType,int>::const_iterator label = leftCounter.begin(); label != leftCounter.end(); label++)
                leftGini += (double)(label->second) * (label->second) / (double)(splitIndex * splitIndex);
            for (typename map<classType,int>::const_iterator label = rightCounter.begin(); label != rightCounter.end(); label++)
                rightGini += (double)(label->second) * (label->second) / (double)((totalSize - splitIndex) * (totalSize - splitIndex));
            return (double)(splitIndex/totalSize) * leftGini + (double)(1 - splitIndex/totalSize) * rightGini;
        }
        int maxFeature(int num) {return int(std::sqrt(num));}
        void splitSamplesVec(const Dataset &dataset,const vector<int> &dataIndex,
            int &featureIndex, double &threshold,
            vector<int> &leftDataIndex,vector<int> &rightDataIndex){
            leftDataIndex.clear();
            rightDataIndex.clear();
            for (vector<int>::const_iterator index = dataIndex.begin(); index != dataIndex.end(); index++){
                if (dataset[*index].getFeatures()[featureIndex] <= threshold)
                    leftDataIndex.push_back(*index);
                else
                    rightDataIndex.push_back(*index);
            }
        }
        void sortByFeatures(const Dataset& dataset,int featureIndex,vector<pair<int, double>>& samplesFeaturesVec) {
            for (vector<pair<int, double>>::iterator sample = samplesFeaturesVec.begin(); sample != samplesFeaturesVec.end(); sample++)
                sample->second = dataset[sample->first].getFeatures()[featureIndex];
            sort(samplesFeaturesVec.begin(), samplesFeaturesVec.end(), 
            [](pair<int,double>& a, pair<int, double>& b) {return a.second < b.second;});
        }
        ProbClass computeTargetProb(const Dataset &dataset,const vector<int> &dataIndex){
            ProbClass resProb;
            for (vector<int>::const_iterator index = dataIndex.begin(); index != dataIndex.end(); index++)
                resProb[dataset[*index].getLabel()]++;
            for (typename ProbClass::iterator prob = resProb.begin(); prob != resProb.end(); prob++)
                prob->second /= dataIndex.size();
            return resProb;
        }
        void chooseBestSplitFeatures(const Dataset &dataset,const vector<int> &dataIndex,int& featureIndex,double& threshold){
            std::set<int> featureBucket;
            vector<int> featuresVec;
            int maxFeatureNum = maxFeature(featureNum);
            while (featureBucket.size() < maxFeatureNum){
                int index = rand() % featureNum;
                featureBucket.insert(index);
            }
            for (std::set<int> ::const_iterator feature = featureBucket.begin(); feature != featureBucket.end(); feature++)
                featuresVec.push_back(*feature);
            int bestFeatureIndex = featuresVec[0];
            double minValue = 1e6, bestThreshold = 0;
            vector<pair<int, double>> samplesFeaturesVec(dataIndex.size());
            for (size_t i = 0; i < dataIndex.size(); i++)
                samplesFeaturesVec[i] = std::make_pair(dataIndex[i],0);
            for (vector<int>::const_iterator feature = featuresVec.begin(); feature != featuresVec.end(); feature++) {
                sortByFeatures(dataset,*feature,samplesFeaturesVec);
                int index = 0;
                for (size_t index = 0 ; index < samplesFeaturesVec.size(); index++){
                    double value = computeGiniIndex(dataset,samplesFeaturesVec,index);
                    if (value <= minValue) {
                        minValue = value;
                        bestThreshold = samplesFeaturesVec[index].second;
                        bestFeatureIndex = *feature;
                    }
                }
            }
            featureIndex = bestFeatureIndex;
            threshold = bestThreshold;                          
        }
        std::shared_ptr<Node> constructNode(const Dataset &dataset,vector<int> dataIndex,int depth){
            ProbClass targetProb = computeTargetProb(dataset, dataIndex);
            bool pureNode = false;
            const double eps = 1e-6;
            for (typename ProbClass::iterator prob = targetProb.begin(); prob != targetProb.end(); prob++)
                if (std::abs(prob->second-1) < eps)
                    pureNode = true;
            if (pureNode || depth >= maxDepth || dataIndex.size() < minSamplesSplit) {
                std::shared_ptr<Node> leaf = std::make_shared<Node>(true);
                leaf->prob = targetProb;
                return leaf;
            }
            int featureIndex;
            double threshold;
            vector<int> leftIndex, rightIndex;
            chooseBestSplitFeatures(dataset,dataIndex,featureIndex,threshold);
            splitSamplesVec(dataset,dataIndex,featureIndex,threshold,leftIndex,rightIndex);
            if ((leftIndex.size() < minSamplesLeaf) or (rightIndex.size() < minSamplesLeaf)) {
                std::shared_ptr<Node> node = std::make_shared<Node>(true,featureIndex,threshold);
                node->prob = targetProb;
                return node;
            } else
                return std::make_shared<Node>(false,featureIndex,threshold,
                    constructNode(dataset,leftIndex,depth+1),constructNode(dataset,rightIndex,depth+1));
        }

    public:
        DecisionTree(int featureNum,int maxDepth,int minSamplesSplit,int minSamplesLeaf)
            :featureNum(featureNum),maxDepth(maxDepth),minSamplesSplit(minSamplesSplit),minSamplesLeaf(minSamplesLeaf){};
        void train(const Dataset &dataset){
            srand(time(0));
            vector<int> dataIndex;
            for (size_t i = 0; i < dataset.size(); i++)
                if (dataset[i].isTrainSample())
                    dataIndex.push_back(i);
            root = constructNode(dataset,dataIndex,0);
        }
        ProbClass predict(const vFloat& x){
            std::shared_ptr<Node> node = root;
            while (!node->isLeaf) {
                if (x[node->featureIndex] <= node->threshold)
                    node = node -> left;
                else
                    node = node -> right;
            }
            return node->prob;
        }
    };
    using DecisionTreeList = vector<std::unique_ptr<DecisionTree>>;
    DecisionTreeList decisionTrees;
    int nEstimators,eachTreeSamplesNum;
    int maxDepth,minSamplesSplit,minSamplesLeaf;
public:
    T_RandomForestClassifier(int nEstimators = 5,int maxDepth = 10, int minSamplesSplit = 2, int minSamplesLeaf = 1, int eachTreeSamplesNum = 1000000)
     : nEstimators(nEstimators),maxDepth(maxDepth),eachTreeSamplesNum(eachTreeSamplesNum),
        minSamplesSplit(minSamplesSplit),minSamplesLeaf(minSamplesLeaf) {
        decisionTrees.reserve(nEstimators);
        this->outputPhotoName = "rf.png";
    }
    ~T_RandomForestClassifier(){}
    virtual void Train(const Dataset& dataset) override{
        this->featureNum = dataset[0].getFeatures().size();
        for (int i = 0; i < nEstimators; i++){
            std::unique_ptr<DecisionTree> tree = std::make_unique<DecisionTree>(this->featureNum,maxDepth,minSamplesSplit,minSamplesLeaf);
            tree->train(dataset);
            decisionTrees.push_back(std::move(tree));
        }
    }
    classType Predict(const vFloat& x) override{
        ProbClass votes;
        for (typename DecisionTreeList::iterator tree = decisionTrees.begin(); tree != decisionTrees.end(); tree++){
            ProbClass singleVote = (*tree)->predict(x);
            for (typename ProbClass::iterator prob = singleVote.begin(); prob != singleVote.end(); prob++)
                votes[prob->first] += prob->second;
        }
        return std::max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
};
#endif