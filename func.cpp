#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "func.hpp"

void StaticPara::InitClassType(Classes ID){
    recordNum = 0;
    classID = ID;
    avg.clear();
    var.clear();
    return;
}
void StaticPara::Sampling(const std::string& entryPath){
    cv::Mat patch = cv::imread(entryPath);
    if (patch.empty()){
        std::cerr << "Image not found!" << std::endl;
        return;
    }
    std::vector<cv::Mat> channels;
    tcb::GenerateFeatureChannels(patch,channels);
    const unsigned int patchRows = patch.rows, patchCols = patch.cols;
    for (unsigned int left = 0; left < patchCols - classifierKernelSize; left+=classifierKernelSize){
        for (unsigned int top = 0; top < patchRows - classifierKernelSize; top+=classifierKernelSize){
            cv::Rect window(left, top, classifierKernelSize, classifierKernelSize);
            vFloat means, vars;
            for (unsigned int i = 0; i < Demisions::dim; i++){
                cv::Mat viewingPatch = channels[i](window);
                cv::Scalar mean, stddev;
                cv::meanStdDev(viewingPatch, mean, stddev);
                means.push_back(mean[0]);
                vars.push_back(stddev[0] * stddev[0]);
            }
            avg.push_back(means);
            var.push_back(vars);
            recordNum++;
        }
    }
    return;
}
float Sample::calcMean(const vFloat& data){
    double sum = 0.0f;
    for (vFloat::const_iterator it = data.begin(); it != data.end(); it++)
        sum += *it;
    return static_cast<float>(sum / data.size());
}
namespace bayes {
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
double NaiveBayesClassifier::CalculateClassProbability(unsigned int classID,const vFloat& x){
    double res = para[static_cast<Classes>(classID)].w;
    for (unsigned int d = 0; d < featureNum; d++){
        float pd = x[d] - para[static_cast<Classes>(classID)].mu[d];
        float vars = para[static_cast<Classes>(classID)].sigma[d] * para[static_cast<Classes>(classID)].sigma[d];
        double exponent = exp(static_cast<double>(- pd * pd / (2 * vars)));
        double normalize = 1.0f / (sqrt(2 * CV_PI) * vars);
        res *= normalize * exponent;
    }
    return res;
}
void NaiveBayesClassifier::Train(const std::vector<Sample>& samples,const float* classProbs){
    featureNum = samples[0].getFeatures().size(); //select all
    para.clear();
    unsigned int classNum = Classes::counter;
    std::vector<double> classifiedFeaturesAvg[classNum],classifiedFeaturesVar[classNum];
    for (int i = 0; i < classNum; i++){
        classifiedFeaturesAvg[i].assign(featureNum,0.0);
        classifiedFeaturesVar[i].assign(featureNum,0.0);
    }
    std::vector<size_t> classRecordNum(classNum,0);
    for (std::vector<Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
        unsigned int label = static_cast<unsigned int>(it->getLabel());
        const vFloat& sampleFeature = it->getFeatures();
        for (unsigned int i = 0; i < featureNum; i++)
            classifiedFeaturesAvg[label][i] += sampleFeature[i];
        classRecordNum[label]++;
    }
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < featureNum; j++)
            classifiedFeaturesAvg[i][j] /= classRecordNum[i];
    for (std::vector<Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
        unsigned int label = static_cast<unsigned int>(it->getLabel());
        const vFloat& sampleFeature = it->getFeatures();
        for (unsigned int i = 0; i < featureNum; i++)
            classifiedFeaturesVar[label][i] += (sampleFeature[i] - classifiedFeaturesAvg[label][i]) * (sampleFeature[i] - classifiedFeaturesAvg[label][i]);
    }
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < featureNum; j++)
            classifiedFeaturesVar[i][j] = std::sqrt(classifiedFeaturesVar[i][j]/classRecordNum[i]);
    for (unsigned int i = 0; i < classNum; i++){
        BasicParaList temp;
        temp.w = classProbs[i];
        temp.mu = classifiedFeaturesAvg[i];
        temp.sigma = classifiedFeaturesVar[i];
        para.push_back(temp);
        {
            std::cout<<"class "<<i<<" counts"<<classRecordNum[i]<<std::endl;
            std::cout<<"w: "<<temp.w<<std::endl;
            std::cout<<"mu: ";
            for (unsigned int j = 0; j < featureNum; j++)
                std::cout<<temp.mu[j]<<" ";
            std::cout<<std::endl;
            std::cout<<"sigma: ";
            for (unsigned int j = 0; j < featureNum; j++)
                std::cout<<temp.sigma[j]<<" ";
            std::cout<<std::endl;
        }
    }
    return;
};
double NonNaiveBayesClassifier::CalculateClassProbability(unsigned int classID,const vFloat& x){
    const unsigned int classNum = Classes::counter;
    double res = log(para[static_cast<Classes>(classID)].w);
    res -= log(determinant(para[classID].convMat))/2;
    vFloat pd = x;
    for (unsigned int d = 0; d < featureNum; d++)
        pd[d] = x[d] - para[static_cast<Classes>(classID)].mu[d];
    vFloat sumX(featureNum, 0.0);
    for (size_t i = 0; i < featureNum; i++)
        for (size_t j = 0; j < featureNum; j++)
            sumX[i] += x[j] * para[classID].invMat[i][j];
    for (unsigned int d = 0; d < featureNum; d++){
        float sum = 0.0f;
        for (unsigned int j = 0; j < featureNum; j++)
            sum += sumX[j] * x[j];
        res -= sum / 2;
    }
    return res;
}
void NonNaiveBayesClassifier::CalcConvMat(fMat convMat,fMat invMat,const std::vector<vFloat>& bucket){
    for (size_t i = 0; i < featureNum; i++){
        convMat[i][i] = CalcConv(bucket[i],bucket[i]) + lambda;
        for (size_t j = i+1; j < featureNum; j++){
            double conv = CalcConv(bucket[i],bucket[j]);
            convMat[i][j] = conv * (1.0f - lambda);
            convMat[j][i] = conv * (1.0f - lambda);
        }
    }
    tcb::CalcInvMat(convMat,invMat,featureNum);
    return;
}
void NonNaiveBayesClassifier::Train(const std::vector<Sample>& samples,const float* classProbs){
    featureNum = samples[0].getFeatures().size(); //select all
    std::cout<<featureNum<<std::endl;
    para.clear();
    para.resize(Classes::counter);
    unsigned int classNum = Classes::counter;
    std::vector<double> classifiedFeaturesAvg[classNum];
    for (int i = 0; i < classNum; i++){
        classifiedFeaturesAvg[i].assign(featureNum,0.0);
        para[i].convMat = new float*[featureNum];
        for (size_t j = 0; j < featureNum; j++)
            para[i].convMat[j] = new float[featureNum];
        para[i].invMat = new float*[featureNum];
        for (size_t j = 0; j < featureNum; j++)
            para[i].invMat[j] = new float[featureNum];
    }
    std::vector<size_t> classRecordNum(classNum,0);
    std::vector<std::vector<vFloat>> sampleBucket(classNum);
    for (unsigned int i = 0; i < classNum; i++)
        sampleBucket[i].resize(featureNum);
    for (std::vector<Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
        unsigned int label = static_cast<unsigned int>(it->getLabel());
        const vFloat& sampleFeature = it->getFeatures();
        for (unsigned int i = 0; i < featureNum; i++){
            classifiedFeaturesAvg[label][i] += sampleFeature[i];
            sampleBucket[label][i].push_back(sampleFeature[i]);
        }
        classRecordNum[label]++;
    }
    for (unsigned int i = 0; i < classNum; i++){
        for (unsigned int j = 0; j < featureNum; j++)
            classifiedFeaturesAvg[i][j] /= classRecordNum[i];
        CalcConvMat(para[i].convMat,para[i].invMat,sampleBucket[i]);
    }
    for (unsigned int i = 0; i < classNum; i++){
        para[i].w = classProbs[i];
        para[i].mu = classifiedFeaturesAvg[i];
    }
    return;
};
void NonNaiveBayesClassifier::LUdecomposition(fMat matrix, fMat L, fMat U){
    for (int i = 0; i < featureNum; i++) { // init LU
        for (int j = 0; j < featureNum; j++) {
            L[i][j] = 0;
            U[i][j] = matrix[i][j];
        }
        L[i][i] = 1;
    }
    for (int i = 0; i < featureNum; i++) { // LU decomposition
        for (int j = i; j < featureNum; j++) 
            for (int k = 0; k < i; ++k) 
                U[i][j] -= L[i][k] * U[k][j];
        for (int j = i + 1; j < featureNum; j++) {
            for (int k = 0; k < i; ++k)
                L[j][i] -= L[j][k] * U[k][i];
            L[j][i] = U[j][i] / U[i][i];
        }
    }
}
double NonNaiveBayesClassifier::determinant(fMat matrix) {
    fMat L = new float*[featureNum];
    fMat U = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++){
        L[i] = new float[featureNum];
        U[i] = new float[featureNum];
        for (size_t j = 0; j < featureNum; j++)
            L[i][j] = U[i][j] = 0;
    }
    LUdecomposition(matrix, L, U);
    double det = 1.0;
    for (int i = 0; i < featureNum; i++)
        det *= U[i][i];
    for (size_t i = 0; i < featureNum; i++){
        delete[] L[i];
        delete[] U[i];
    }
    delete[] L;
    delete[] U;
    return det;
}
NonNaiveBayesClassifier::~NonNaiveBayesClassifier(){
    for (std::vector<convParaList>::const_iterator it = para.begin(); it != para.end(); it++){
        if(it->convMat != nullptr){
            for(size_t i = 0;i < featureNum;i++)
                delete[] it->convMat[i];
            delete[] it->convMat;
        }
        if(it->invMat != nullptr){
            for(size_t i = 0;i < featureNum;i++)
                delete[] it->invMat[i];
            delete[] it->invMat;
        }
    }
}
};//namespace bayes
namespace linear{
FisherClassifier::~FisherClassifier(){
    if (projMat != nullptr){
        for (size_t i = 0; i < Classes::counter; i++)
            delete[] projMat[i];
        delete[] projMat;
    }
}
void FisherClassifier::Train(const std::vector<Sample>& samples){
    using namespace Eigen;
    featureNum = samples[0].getFeatures().size(); //select all
    fMat Sw,Sb;
    Sw= new float*[featureNum];
    Sb = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++){
        Sw[i] = new float[featureNum];
        Sb[i] = new float[featureNum];
        for (size_t j = 0; j < featureNum; j++)
            Sw[i][j] = 0.0f,Sb[i][j] = 0.0f;
    }

    CalcSwSb(Sw,Sb,samples);

    float** invSw = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++){
        invSw[i] = new float[featureNum];
        for (size_t j = 0; j < featureNum; j++)
            invSw[i][j] = 0.0f;
    }
    tcb::CalcInvMat(Sw,invSw,featureNum);

    //std::vector<vFloat> featureMat(featureNum,vFloat(featureNum,0.0f));
    MatrixXd featureMat(featureNum,featureNum);
    for (int i = 0; i< featureNum; i++)
        for (int j = 0; j < featureNum; j++)
            for (int k = 0; k < featureNum; k++)
                //featureMat[i][j] += invSw[i][k] * Sb[k][j];
                featureMat(i,j) += invSw[i][k] * Sb[k][j];

    /*
    vFloat EigVal(featureNum,0.0f);
    std::vector<vFloat> EigVec(featureNum,vFloat(featureNum,0.0f));
    tcb::CalcEigen(featureMat,EigVal,EigVec,featureNum);
    */
    
    SelfAdjointEigenSolver<MatrixXd> eig(featureMat);
    VectorXd EigVal = eig.eigenvalues().real();
    MatrixXd EigVec = eig.eigenvectors().real();
    std::cout<<"Eigenvalues: "<<std::endl;
    for (int i = 0; i < featureNum; i++)
        std::cout<<EigVal(i)<<' ';
    std::cout<<std::endl;
    std::cout<<"Eigenvectors: "<<std::endl;
    for (int i = 0; i < featureNum; i++){
        for (int j = 0; j < featureNum; j++)
            std::cout<<EigVec(i,j)<<' ';
        std::cout<<std::endl;
    }
    

    const unsigned int classNum = Classes::counter;
    projMat = new float*[classNum];
    for (size_t i = 0; i < classNum; i++){
        projMat[i] = new float[featureNum];
        for (size_t j = 0; j < featureNum; j++){
            //projMat[i][j] = EigVec[i][j];
            projMat[i][j] = EigVec(i,j);
            std::cout<<projMat[i][j]<<' ';
        }
        std::cout<<std::endl;
    }
    for(size_t i = 0;i < featureNum;i++){
        delete[] Sw[i];
        delete[] Sb[i];
        delete[] invSw[i];
    }
    delete[] Sw;
    delete[] Sb;
    delete[] invSw;
    return;
}
Classes FisherClassifier::Predict(const vFloat& x){
    Classes resClass = Classes::Unknown;
    double maxProb = -10e9;
    for (unsigned int classID = 0; classID < Classes::counter; classID++){
        double prob = 0.0f;
        for (unsigned int i = 0; i < featureNum; i++)
            prob += x[i] * projMat[classID][i];
        if (prob > maxProb){
            maxProb = prob;
            resClass = static_cast<Classes>(classID);
        }
    }
    return resClass;
}
void FisherClassifier::CalcSwSb(float** Sw,float** Sb,const std::vector<Sample>& samples){
    unsigned int classNum = Classes::counter;
    vFloat featureAvg(featureNum, 0.0f);

    std::vector<double> classifiedFeaturesAvg[classNum];
    for (int i = 0; i < classNum; i++)
        classifiedFeaturesAvg[i].assign(featureNum,0.0);
    std::vector<size_t> classRecordNum(classNum,0);
    for (std::vector<Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
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

    for (std::vector<Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
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
    return;
}
bool LinearClassify(const cv::Mat& rawimage,FisherClassifier* classifer,std::vector<std::vector<Classes>>& patchClasses){
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
            tcb::GenerateFeatureChannels(sample, channels);
            tcb::CalcChannelMeanStds(channels, data);
            rowClasses.push_back(classifer->Predict(data));
        }
        patchClasses.push_back(rowClasses);
    }
    return true;
}
};//namespace linear