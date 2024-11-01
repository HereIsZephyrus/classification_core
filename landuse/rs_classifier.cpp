#include "rs_classifier.hpp"
namespace weilaicheng{
template<>
void land_StaticPara::InitClassType(LandCover ID){
    recordNum = 0;
    classID = ID;
    avg.clear();
    var.clear();
    return;
}
template<>
void land_StaticPara::Sampling(const std::string& entryPath){
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
            for (unsigned int i = 0; i < Spectra::spectralNum; i++){
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
template<>
double land_NaiveBayesClassifier::CalculateClassProbability(unsigned int classID,const vFloat& x){
    double res = para[static_cast<LandCover>(classID)].w;
    for (unsigned int d = 0; d < featureNum; d++){
        float pd = x[d] - para[static_cast<LandCover>(classID)].mu[d];
        float vars = para[static_cast<LandCover>(classID)].sigma[d] * para[static_cast<LandCover>(classID)].sigma[d];
        double exponent = exp(static_cast<double>(- pd * pd / (2 * vars)));
        double normalize = 1.0f / (sqrt(2 * CV_PI) * vars);
        res *= normalize * exponent;
    }
    return res;
}
template<>
void land_NaiveBayesClassifier::Train(const std::vector<land_Sample>& samples,const float* classProbs){
    featureNum = samples[0].getFeatures().size(); //select all
    para.clear();
    unsigned int classNum = getClassNum();
    std::vector<double> classifiedFeaturesAvg[classNum],classifiedFeaturesVar[classNum];
    for (int i = 0; i < classNum; i++){
        classifiedFeaturesAvg[i].assign(featureNum,0.0);
        classifiedFeaturesVar[i].assign(featureNum,0.0);
    }
    std::vector<size_t> classRecordNum(classNum,0);
    for (std::vector<land_Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
        unsigned int label = static_cast<unsigned int>(it->getLabel());
        const vFloat& sampleFeature = it->getFeatures();
        for (unsigned int i = 0; i < featureNum; i++)
            classifiedFeaturesAvg[label][i] += sampleFeature[i];
        classRecordNum[label]++;
    }
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < featureNum; j++)
            classifiedFeaturesAvg[i][j] /= classRecordNum[i];
    for (std::vector<land_Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
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
size_t land_NaiveBayesClassifier::getClassNum() const{return LandCover::CoverType;}
template<>
size_t land_FisherClassifier::getClassNum() const{return LandCover::CoverType;}
template<>
void land_FisherClassifier::CalcSwSb(float** Sw,float** Sb,const std::vector<land_Sample>& samples){
    unsigned int classNum = getClassNum();
    vFloat featureAvg(this->featureNum, 0.0f);
    std::vector<double> classifiedFeaturesAvg[classNum];
    for (int i = 0; i < classNum; i++)
        classifiedFeaturesAvg[i].assign(this->featureNum,0.0);
    std::vector<size_t> classRecordNum(classNum,0);
    for (std::vector<land_Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
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

    for (std::vector<land_Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
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
void land_FisherClassifier::Train(const std::vector<land_Sample>& samples){
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
Classes land_FisherClassifier::Predict(const vFloat& x){
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
}