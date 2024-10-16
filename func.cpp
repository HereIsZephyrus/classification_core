#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "func.hpp"

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
//const float bayes::NaiveBayesClassifier::lambda = 0.1f;
namespace tcb{
bool TopHat(cv::Mat &image,int xSize,int ySize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(xSize, ySize));
    cv::morphologyEx(image, image, cv::MORPH_OPEN, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Top Hat Transform", image);
        cv::waitKey(0);
        cv::destroyWindow("Top Hat Transform");
    }
    return true;
}
bool BottomHat(cv::Mat &image,int xSize,int ySize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(xSize, ySize));
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Bottom Hat Transform", image);
        cv::waitKey(0);
        cv::destroyWindow("Bottom Hat Transform");
    }
    return true;
}
bool GaussianSmooth(cv::Mat &image,int xSize,int ySize,int sigma){
    cv::GaussianBlur(image, image, cv::Size(xSize, ySize), 0);
    if (SHOW_WINDOW){
        cv::imshow("Smoothed Image with Gaussian Blur", image);
        cv::waitKey(0);
        cv::destroyWindow("Smoothed Image with Gaussian Blur");
    }
    return true;
}
bool rgb2gray(cv::Mat &image){
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    if (SHOW_WINDOW){
        cv::imshow("Gray Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Gray Image");
    }
    return true;
}
bool Sobel(cv::Mat &image,int dx,int dy,int bandwidth){
    cv::Mat sobeled;
    cv::Sobel(image, sobeled, CV_64F, dx, dy, bandwidth);
    cv::convertScaleAbs(sobeled, image);
    if (SHOW_WINDOW){
        cv::imshow("Sobel Image (Vertical Lines)", image);
        cv::waitKey(0);
        cv::destroyWindow("Sobel Image (Vertical Lines)");
    }
    return true;
}

bool Laplacian(cv::Mat &image,int bandwidth){
    cv::Mat laplacianImage;
    cv::Laplacian(image, laplacianImage, CV_16S, bandwidth);
    cv::Mat absLaplacianImage;
    cv::convertScaleAbs(laplacianImage, image);
    if (SHOW_WINDOW){
        cv::imshow("Laplacian Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Laplacian Image");
    }
    return true;
}

bool BoxSmooth(cv::Mat &image,int xSize,int ySize){
    cv::blur(image, image, cv::Size(xSize, ySize));
    if (SHOW_WINDOW){
        cv::imshow("Smoothed Image with Box Blur", image);
        cv::waitKey(0);
        cv::destroyWindow("Smoothed Image with Box Blur");
    }
    return true;
}
bool Erode(cv::Mat &image,int kernelSize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::erode(image, image, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Eroded Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Eroded Image");
    }
    return true;
}
bool Dilate(cv::Mat &image,int kernelSize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::dilate(image, image, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Dilated Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Dilated Image");
    }
    return true;
}
bool drawCircleDDA(cv::Mat &image, int h, int k, float rx,float ry) {
    int lineThickness = 8;
    for (int theta = 0; theta < 3600; theta++) {
        int x = static_cast<int>(h + rx * cos(theta/10 * CV_PI / 180));
        int y = static_cast<int>(k + ry * sin(theta/10 * CV_PI / 180));
        for (int i = -lineThickness; i <= lineThickness; i++)
            for (int j = -lineThickness; j <= lineThickness; j++)
                if (x + i >= 0 && x + i < image.cols && y + j >= 0 && y + j < image.rows)
                    image.at<uchar>(y+j,x+i) = 255;
    }
    return true;
}
bool GenerateFeatureChannels(const cv::Mat &image,std::vector<cv::Mat> &channels){
    std::vector<cv::Mat> HSVchannels;
    channels.clear();
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
    cv::split(hsvImage, HSVchannels);
    channels = HSVchannels;
    cv::Mat sobelx,sobely,magnitude,angle;
    cv::Sobel(image, sobelx, CV_64F, 1, 0, 3);
    cv::Sobel(image, sobely, CV_64F, 0, 1, 3);
    cv::cartToPolar(sobelx, sobely, magnitude, angle, true);
    //channels.push_back(magnitude);
    channels.push_back(angle);
    return true;
}
bool CalcChannelMeanStds(const std::vector<cv::Mat> & channels, vFloat & data){
    data.clear();
    for (std::vector<cv::Mat>::const_iterator it = channels.begin(); it != channels.end(); it++){
        cv::Scalar mean, stddev;
        cv::meanStdDev(*it, mean, stddev);
        data.push_back(cv::mean(*it)[0]);
        data.push_back(stddev[0] * stddev[0]);
    }
    return true;
}
bool CalcInvMat(float** const convMat,float ** invMat,int num){
    fMat augmented = new float*[num];
    for (size_t i = 0; i < num; i++)
        augmented[i] = new float[num*2];
    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < num; j++) 
            augmented[i][j] = convMat[i][j];
        for (size_t j = num; j < 2 * num; j++)
            if (i == (j - num))
                augmented[i][j] = 1.0;
            else
                augmented[i][j] = 0.0;
    }
    for (size_t i = 0; i < num; i++) {
        double pivot = augmented[i][i];
        for (size_t j = 0; j < 2 * num; j++)
            augmented[i][j] /= pivot;
        for (size_t k = 0; k < num; ++k) {
            if (k == i) 
                continue;
            double factor = augmented[k][i];
            for (size_t j = 0; j < 2 * num; j++) 
                augmented[k][j] -=  factor * augmented[i][j];
        }
    }
    for (size_t i = 0; i < num; i++)
        for (size_t j = 0; j < num; j++)
            invMat[i][j] = augmented[i][j + num];
    if (augmented != nullptr){
        for(size_t i = 0;i < num;i++)
            delete[] augmented[i];
        delete[] augmented;
    }
    return true;
}
bool CalcEigen(const std::vector<vFloat>& matrix, vFloat& eigVal, std::vector<vFloat>& eigVec, const int num){
    std::vector<vFloat> Q(num,vFloat(num,0.0f));
    std::vector<vFloat> R(num,vFloat(num,0.0f));
    std::vector<vFloat> A(num,vFloat(num,0.0f));
    for (int i = 0; i < num; i++)
        eigVec[i][i] = 1.0f;
    A = matrix;
    const int maxIter = 3;
    const float tolerance = 1e-6;
    for (int iter = 0; iter < maxIter; iter++){
        //QR decomposition
        for (int j = 0; j < num; j++) {
            for (int i = 0; i < num; i++)
                Q[i][j] = A[i][j];
            for (int i = 0; i < j; ++i) {
                float temp = 0;
                for (size_t k = 0; k < num; ++k)
                    temp += Q[k][j] * A[k][i];
                R[i][j] = temp;
                for (int k = 0; k < num; ++k)
                    Q[k][j] -= R[i][j] * Q[k][i];
            }
            {
                float temp = 0;
                for (size_t i = 0; i < num; ++i)
                    temp += Q[i][j] * Q[i][j];
                R[j][j] = sqrt(temp);
            }
            for (int i = 0; i < num; ++i)
                Q[i][j] /= R[j][j];
        }

        //update A by QR
        //std::cout<<"iter "<<iter<<std::endl;
        for (int i = 0; i < num; i++){
            for (int j = 0; j < num; j++) {
                A[i][j] = 0.0f;
                for (int k = 0; k < num; k++)
                    A[i][j] += R[k][j] * Q[i][k];
                //std::cout<<A[i][j]<<' ';
            }
            //std::cout<<std::endl;
        }
        
        //update eigVec by Q
        std::vector<vFloat> temp(num,vFloat(num,0.0f));
        for (int i = 0; i < num; i++)
            for (int j = 0; j < num; j++)
                for (int k = 0; k < num; k++){
                    temp[i][j] += eigVec[i][k] * Q[k][j];
                    //std::cout<<temp[i][j]<<"+="<<eigVec[i][k]<<'*'<<Q[k][j]<<' ';
                }
        eigVec = temp;
        for (int i = 1; i < num; i++)
            if (fabs(A[i][i - 1]) < tolerance)
                break;
    }
    for (int i = 0; i < num; i++)
        eigVal[i] = A[i][i];
    return true;
}
};//namespace tcb

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