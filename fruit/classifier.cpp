#include "classifier.hpp"
using namespace Eigen;
std::string classFolderNames[Classes::counter] = 
{"desk","apple","blackplum","dongzao","grape","peach","yellowpeach"};
std::unordered_map<Classes,cv::Scalar> classifyColor = {
    {Classes::Desk,cv::Scalar(0,0,0)}, // black
    {Classes::Apple,cv::Scalar(0,0,255)}, // red
    {Classes::Blackplum,cv::Scalar(255,0,255)}, // magenta
    {Classes::Dongzao,cv::Scalar(42,42,165)}, // brzone,
    {Classes::Grape,cv::Scalar(0,255,0)}, // green
    {Classes::Peach,cv::Scalar(203,192,255)}, // pink
    {Classes::Yellowpeach,cv::Scalar(0,255,255)}, // yellow    
    {Classes::Edge,cv::Scalar(255,255,255)}, // white
    {Classes::Unknown,cv::Scalar(211,211,211)}// gray
};
template<>
void StaticPara::InitClassType(Classes ID){
    recordNum = 0;
    classID = ID;
    avg.clear();
    var.clear();
    return;
}
template<>
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
namespace bayes {
template<>
size_t NaiveBayesClassifier::getClassNum() const{return Classes::counter;}
template<>
size_t NonNaiveBayesClassifier::getClassNum() const{return Classes::counter;}
template<>
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
template<>
void NaiveBayesClassifier::Train(const std::vector<Sample>& samples,const float* classProbs){
    featureNum = samples[0].getFeatures().size(); //select all
    para.clear();
    unsigned int classNum = getClassNum();
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
}
template<>
double NonNaiveBayesClassifier::CalculateClassProbability(unsigned int classID,const vFloat& x){
    const unsigned int classNum = getClassNum();
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
template<>
void NonNaiveBayesClassifier::Train(const std::vector<Sample>& samples,const float* classProbs){
    featureNum = samples[0].getFeatures().size(); //select all
    //std::cout<<featureNum<<std::endl;
    para.clear();
    para.resize(getClassNum());
    unsigned int classNum = getClassNum();
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
};//namespace bayes
namespace linear{
template<>
size_t FisherClassifier::getClassNum() const{return Classes::counter;}
template<>
void FisherClassifier::CalcSwSb(float** Sw,float** Sb,const std::vector<T_Sample<Classes>>& samples){
    unsigned int classNum = getClassNum();
    vFloat featureAvg(this->featureNum, 0.0f);
    std::vector<double> classifiedFeaturesAvg[classNum];
    for (int i = 0; i < classNum; i++)
        classifiedFeaturesAvg[i].assign(this->featureNum,0.0);
    std::vector<size_t> classRecordNum(classNum,0);
    for (std::vector<T_Sample<Classes>>::const_iterator it = samples.begin(); it != samples.end(); it++){
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

    for (std::vector<T_Sample<Classes>>::const_iterator it = samples.begin(); it != samples.end(); it++){
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
    return;
}
template<>
void FisherClassifier::Train(const std::vector<Sample>& samples){
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
    
    const unsigned int classNum = getClassNum();
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
template<>
Classes FisherClassifier::Predict(const vFloat& x){
    Classes resClass = Classes::Unknown;
    double maxProb = -10e9;
    for (unsigned int classID = 0; classID < getClassNum(); classID++){
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