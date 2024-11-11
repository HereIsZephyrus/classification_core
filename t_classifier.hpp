#ifndef TCLASSIFIER_HPP
#define TCLASSIFIER_HPP
#include <cstring>
#include <string>
#include <map>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "func.hpp"

constexpr int defaultClassifierKernelSize = 9;
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
    using SampleList = std::vector<T_Sample<classType>>;
    virtual classType Predict(const vFloat& x) = 0;
    virtual size_t getClassNum() const{return 0;}
    const std::string& printPhoto() const{return outputPhotoName;}
    void Examine(const std::vector<T_Sample<classType>>& samples){
        size_t pcorrectNum = 0, rcorrectNum = 0;
        for (typename SampleList::const_iterator it = samples.begin(); it != samples.end(); it++){
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
    void Classify(const cv::Mat& featureImage,std::vector<std::vector<classType>>& pixelClasses,int classifierKernelSize = defaultClassifierKernelSize){
        using ClassMat = std::vector<std::vector<classType>>;
        using vClasses = std::vector<classType>;
        int rows = featureImage.rows, cols = featureImage.cols;
        for (int r = classifierKernelSize/2; r <= rows - classifierKernelSize; r+=classifierKernelSize/2){
            vClasses rowClasses;
            bool lastRowCheck = (r >= (rows - classifierKernelSize));
            for (int c = classifierKernelSize/2; c <= cols - classifierKernelSize; c+=classifierKernelSize/2){
                bool lastColCheck = (c >= (cols - classifierKernelSize));
                cv::Rect window(c - classifierKernelSize/2 ,r - classifierKernelSize/2,  classifierKernelSize - lastColCheck, classifierKernelSize - lastRowCheck);  
                cv::Mat sample = featureImage(window);
                std::vector<cv::Mat> channels;
                vFloat data;
                rowClasses.push_back(classifer->Predict(data));
            }
            patchClasses.push_back(rowClasses);
        }
        ClassMat::const_iterator row = patchClasses.begin();
        { //tackle the first line
            vClasses temprow;
            for (vClasses::const_iterator col = row->begin(); col != row->end(); col++){
                if (col == row->begin()){
                    temprow.push_back(*col);
                    continue;
                }
                if ((*col) != (*(col-1)))//horizontalEdgeCheck
                    temprow.push_back(Classes::Edge);
                else
                    temprow.push_back(*col);
                if ((col+1) == row->end())// manually add the last element
                    temprow.push_back(*col);
            }
            pixelClasses.push_back(temprow);
        }
        vClasses::const_iterator lastRowBegin = row->begin();
        row++;
        for (; row != patchClasses.end(); row++){
            vClasses temprow;
            vClasses::const_iterator col = row->begin(),diagonalCol = lastRowBegin;
            { //tackle the first col
                if (*col == *diagonalCol)
                    temprow.push_back(*col);
                else
                    temprow.push_back(Classes::Edge);
                col++;
            }
            for (;col != row->end(); col++,diagonalCol++){
                bool horizontalEdgeCheck = (*col) == (*(col-1));
                bool verticalEdgeCheck = (*col) == (*(diagonalCol+1));
                bool diagonalEdgeCheck = (*col) == (*(diagonalCol));
                if (horizontalEdgeCheck && verticalEdgeCheck && diagonalEdgeCheck)
                    temprow.push_back(*col);
                else
                    temprow.push_back(Classes::Edge);
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
            vClasses::const_iterator col = row->begin(),diagonalCol = lastRowBegin;
            { //tackle the first col
                if (*col == *diagonalCol)
                    temprow.push_back(*col);
                else
                    temprow.push_back(Classes::Edge);
                col++;
            }
            for (;col != row->end(); col++,diagonalCol++){
                bool horizontalEdgeCheck = (*col) == (*(col-1));
                bool verticalEdgeCheck = (*col) == (*(diagonalCol+1));
                bool diagonalEdgeCheck = (*col) == (*(diagonalCol));
                if (horizontalEdgeCheck && verticalEdgeCheck && diagonalEdgeCheck)
                    temprow.push_back(*col);
                else
                    temprow.push_back(Classes::Edge);
                if ((col+1) == row->end())// manually add the last element
                    temprow.push_back(*col);
            }
            pixelClasses.push_back(temprow);
        }
        return true;
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
    virtual void Train(const std::vector<T_Sample<classType>>& samples,const float* classProbs) = 0;
};
template <typename classType>
class T_NaiveBayesClassifier : public T_BayesClassifier<BasicParaList,classType>{
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x){
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
};
template <typename classType>
class T_NonNaiveBayesClassifier : public T_BayesClassifier<convParaList,classType>{
protected:
    double CalculateClassProbability(unsigned int classID,const vFloat& x){
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
        using SampleList = std::vector<T_Sample<classType>>;
        unsigned int classNum = this->getClassNum();
        vFloat featureAvg(this->featureNum, 0.0f);
        std::vector<double> classifiedFeaturesAvg[classNum];
        for (int i = 0; i < classNum; i++)
            classifiedFeaturesAvg[i].assign(this->featureNum,0.0);
        std::vector<size_t> classRecordNum(classNum,0);
        for (typename SampleList::const_iterator it = samples.begin(); it != samples.end(); it++){
            unsigned int label = static_cast<unsigned int>(it->getLabel());
            const vFloat& sampleFeature = it->getFeatures();
            for (unsigned int i = 0; i < this->featureNum; i++)
                classifiedFeaturesAvg[label][i] += sampleFeature[i];
            classRecordNum[label]++;
        }
        for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < this->featureNum; j++){
            classifiedFeaturesAvg[i][j] /= classRecordNum[i];
            featureAvg[j] += classifiedFeaturesAvg[i][j];
        }
        for (unsigned int j = 0; j < this->featureNum; j++)
            featureAvg[j] /= classNum;

        for (typename SampleList::const_iterator it = samples.begin(); it != samples.end(); it++){
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
    T_FisherClassifier() {projMat = nullptr;this->outputPhotoName = "fisher.png";}
    ~T_FisherClassifier(){
        if (projMat != nullptr){
            for (size_t i = 0; i < this->getClassNum(); i++)
                delete[] projMat[i];
            delete[] projMat;
        }
    }
    virtual void Train(const std::vector<SampleType>& samples) = 0;
    classType Predict(const vFloat& x){
        classType resClass;
        double maxProb = -10e9;
        for (unsigned int classID = 0; classID < this->getClassNum(); classID++){
            double prob = 0.0f;
            for (unsigned int i = 0; i < this->featureNum; i++)
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
protected:
    double bias;
    vFloat weights;
    double learningRate;
    int maxIter;
    constexpr double regularizationParam = 0.01;
    double weightProduct(const std::vector<int>& vec) {
        double result = 0.0;
        for (size_t i = 0; i < weights.size(); i++) {
            result += weights[i] * vec[i];
        }
        return result;
    }
    class OVOSVM {
    public:
        OVOSVM(classType pos,classType neg,double learningRate = 0.8, int maxIter = 1000)
        : learningRate(learningRate),maxIter(maxIter),positiveClass(pos),negetiveClass(neg) {}
        void fit(const std::vector<vFloat>& X, const std::vector<int>& y) {
            int sampleNum = X.size();
            int featureNum = X[0].size();
            // Initialize weights and bias
            weights.assign(featureNum, 0.0);
            bias = 0.0;
            // Training the SVM
            for (int iter = 0; iter < maxIter; ++iter) {
                for (int i = 0; i < sampleNum; i++) {
                    double linear_output = weighting(weights, X[i]) + bias;
                    if (y[i] * linear_output < 1) {
                        // Update weights and bias
                        for (int j = 0; j < featureNum; ++j)
                            weights[j] += learningRate * (y[i] * X[i][j] - 2 * regularizationParam * weights[j]);
                        bias += learningRate * y[i];
                    } else {
                        // Regularization
                        for (int j = 0; j < featureNum; ++j)
                            weights[j] -= learningRate * 2 * regularizationParam * weights[j];
                    }
                }
            }
        }
        bool predict(const vFloat& sample) const{return weighting(weights, sample) + bias;}
        classType getPositiveClass() const{ return positiveClass; }
        classType getNegetiveClass() const{ return negetiveClass; }
    private:
        double learningRate;
        int maxIter;
        double bias;
        vFloat weights;
        double regularizationParam = 0.01;
        classType positiveClass,negetiveClass;
        double weighting(const vFloat& vec) {
            double result = 0.0;
            for (size_t i = 0; i < vec.size(); i++) {
                result += weights[i] * vec[i];
            }
            return result;
        }
    };
    std::vector<std::unique_ptr<OVOSVM>> classifiers;
public:
    T_SVMClassifier(double rate = 0.8f, int iter = 1000):learningRate(rate),maxIter(iter) {this->outputPhotoName = "svm.png";}
    ~T_SVMClassifier(){}
    virtual void Train(const std::vector<T_Sample<classType>>& samples) = 0;
    classType Predict(const vFloat& x){
        std::vector<int> classVote[getClassNum()];
        classVote.assign(getClassNum(),0);
        for (std::vector<OVOSVM>::iterator it = classifiers.begin(); it != classifiers.end(); it++){
            if (it->getNegtiveClass() > getClassNum()){
                if (it->predict(x))
                    return it->getPositiveClass();
            }else{
                if (it->predict(x))
                    ++classVote[static_cast<unsigned int>(it->getPositiveClass())];
                else
                    ++classVote[static_cast<unsigned int>(it->getNegetiveClass())];
            }
        }
        int maxVote = 0;
        classType resClass;
        for (std::vector<int>::const_iterator it = classVote.begin(); it != classVote.end(); it++)
            if (*it > maxVote){
                maxVote = *it;
                resClass = static_cast<classType>(it - classVote.begin());
            }
        return resClass;
    }
};
template <class classType>
class T_BPClassifier : public T_Classifier<classType>{
protected:
    int classNum;
    vFloat weights;
    double learningRate;
    vFloat forward(const vFloat& inputs) {
        float output = 0;
        for (int i = 0; i < this->featureNum; i++)
            output += inputs[i] * weights[i];
        output = activation(output);
        return output;
    }
    void backward(const vFloat& inputs, classType targets) {
        float output = forward(inputs);
        float errors = static_cast<float>(targets) - outputs;
        for (int i = 0; i < this->featureNum; i++)
            weights[i] += learningRate * errors * inputs[i];
    }
    double activation(double x) {return 1.0 / (1.0 + exp(-x));}
public:
    T_BPClassifier(double rate = 0.3):classNum(0),learningRate(rate) {this->outputPhotoName = "bp.png";}
    ~T_BPClassifier(){}
    virtual void Train(const std::vector<T_Sample<classType>>& samples) = 0;
    classType Predict(const vFloat& x){
        float res = 0;
        for (int i = 0; i < this->featureNum; i++)
            res += x[i] * weights[i];
        return static_cast<classType>((unsigned int)(res + 0.5));
    }
};
template <class classType>
class T_RandomForestClassifier : public T_Classifier<classType>{
protected:
    class DecisionTree {
    public:
        DecisionTree(int maxDepth) : maxDepth(maxDepth) {}
        void fit(const std::vector<vFloat>& X, const std::vector<int>& y) {
            this->X = X;
            this->y = y;
            root = buildTree(0);
        }
        classType predict(const vFloat& sample) {return traverseTree(root, sample);}
    private:
        struct Node {
            int featureIndex = -1;
            double threshold = 0;
            classType label;
            Node* left = nullptr;
            Node* right = nullptr;
        };
        Node* root;
        int maxDepth;
        std::vector<vFloat> X;
        std::vector<classType> y;
        Node* buildTree(int depth) {
            // Base case: if all labels are the same or max depth reached
            if (depth >= maxDepth || std::all_of(y.begin(), y.end(), [&](classType label) { return label == y[0]; }))
                return new Node{ -1, 0, y[0] }; // Return a leaf node
            // Find the best split
            int bestFeature = -1;
            double bestThreshold = 0;
            int bestGain = -1;
            for (size_t featureIndex = 0; featureIndex < X[0].size(); featureIndex++){
                // Collect potential thresholds
                vFloat thresholds;
                for (const std::vector<vFloat>::iterator sample = X.begin(); sample != X.end(); sample++)
                    thresholds.push_back((*sample)[featureIndex]);
                std::sort(thresholds.begin(), thresholds.end());
                thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
                for (vFloat::iterator threshold = thresholds.begin(); threshold != thresholds.end(); threshold++) {
                    std::vector<int> leftLabels, rightLabels;
                    for (size_t i = 0; i < X.size(); ++i) {
                        if (X[i][featureIndex] <= *threshold)
                            leftLabels.push_back(y[i]);
                        else
                            rightLabels.push_back(y[i]);
                    }
                    int gain = calculateGain(leftLabels, rightLabels);
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = featureIndex;
                        bestThreshold = *threshold;
                    }
                }
            }
            if (bestGain == -1)
                return new Node{ -1, 0, y[0] }; // Create a leaf node
            // Split datasets
            std::vector<vFloat> leftX, rightX;
            std::vector<int> leftY, rightY;
            for (size_t i = 0; i < X.size(); ++i)
                if (X[i][bestFeature] <= bestThreshold) {
                    leftX.push_back(X[i]);
                    leftY.push_back(y[i]);
                } else {
                    rightX.push_back(X[i]);
                    rightY.push_back(y[i]);
                }
            Node* node = new Node{ bestFeature, bestThreshold, -1 };
            node->left = buildTree(depth + 1);
            node->right = buildTree(depth + 1);
            return node;
        }
        classType traverseTree(Node* node, const vFloat& sample) {
            if (node->label != -1)
                return node->label; // Leaf node
            if (sample[node->featureIndex] <= node->threshold)
                return traverseTree(node->left, sample);
            else
                return traverseTree(node->right, sample);
        }
        int calculateGain(const std::vector<int>& leftLabels, const std::vector<int>& rightLabels) {
            // Implement simple gain calculation (Gini impurity or entropy)
            int leftSize = leftLabels.size();
            int rightSize = rightLabels.size();
            int totalSize = leftSize + rightSize;
            if (totalSize == 0) return 0;
            double leftImpurity = calculateImpurity(leftLabels);
            double rightImpurity = calculateImpurity(rightLabels);
            return (leftImpurity * leftSize + rightImpurity * rightSize);
        }
        double calculateImpurity(const std::vector<int>& labels) {
            std::map<int, int> counts;
            for (int label : labels) {
                counts[label]++;
            }
            double impurity = 1.0;
            for (const auto& count : counts) {
                double prob = static_cast<double>(count.second) / labels.size();
                impurity -= prob * prob;
            }
            return impurity;
        }
    };
    int nTrees;
    int maxDepth;
    std::vector<DecisionTree> trees;
public:
    T_RandomForestClassifier(int nTrees, int maxDepth) : nTrees(nTrees), maxDepth(maxDepth) {this->outputPhotoName = "rf.png";}
    ~T_RandomForestClassifier(){}
    virtual void Train(const std::vector<T_Sample<classType>>& samples) = 0;
    classType Predict(const vFloat& x){
        std::map<classType, int> votes;
        for (std::vector<DecisionTree>::const_iterator tree = trees.begin(); tree != trees.end(); tree++){
            classType label = tree->predict(sample);
            votes[label]++;
        }
        return std::max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
};
#endif