#include <cmath>
#include <algorithm>
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
};//namespace tcb
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
    res /= n;
    return res;
}
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
/*
Classes NaiveBayesClassifier::Predict(const vFloat& x){
    double maxProb = -1.0f;
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
Classes NonNaiveBayesClassifier::Predict(const vFloat& x){
    double maxProb = -1.0f;
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
*/
void NaiveBayesClassifier::Train(const std::vector<Sample>& samples,const float* classProbs){
    size_t featureNum = samples[0].getFeatures().size(); //select all
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
    for (size_t i = 0; i < featureNum; ++i)
        for (size_t j = 0; j < featureNum; ++j)
            sumX[i] += x[j] * para[classID].invMat[i][j];
    for (unsigned int d = 0; d < featureNum; d++){
        float sum = 0.0f;
        for (unsigned int j = 0; j < featureNum; j++)
            sum += sumX[j] * x[j];
        res -= sum / 2;
    }
    return res;
}
void NonNaiveBayesClassifier::CalcConvMat(float** convMat,float** invMat,const std::vector<vFloat>& bucket){
    //const unsigned int classNum = Classes::counter;
    convMat = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++)
        convMat[i] = new float[featureNum];
    for (size_t i = 0; i < featureNum; i++){
        convMat[i][i] = 1.0f;
        for (size_t j = i+1; j < featureNum; j++){
            double conv = CalcConv(bucket[i],bucket[j]);
            convMat[i][j] = conv * (1.0f - lambda);
            convMat[j][i] = conv * (1.0f - lambda);
        }
    }
    float** augmented = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++)
        augmented[i] = new float[featureNum*2];
    for (size_t i = 0; i < featureNum; i++) {
        for (size_t j = 0; j < featureNum; j++) 
            augmented[i][j] = convMat[i][j];
        for (size_t j = featureNum; j < 2 * featureNum; ++j)
            if (i == (j - featureNum))
                augmented[i][j] = 1.0;
            else
                augmented[i][j] = 0.0;
    }
    for (size_t i = 0; i < featureNum; i++) {
        double pivot = augmented[i][i];
        for (size_t j = 0; j < 2 * featureNum; j++)
            augmented[i][j] /= pivot;
        for (size_t k = 0; k < featureNum; ++k) {
            if (k == i) 
                continue;
            double factor = augmented[k][i];
            for (size_t j = 0; j < 2 * featureNum; ++j) 
                augmented[k][j] -=  factor * augmented[i][j];
        }
    }
    invMat = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++)
        invMat[i] = new float[featureNum];
    for (size_t i = 0; i < featureNum; i++)
        for (size_t j = 0; j < featureNum; j++)
            invMat[i][j] = augmented[i][j + featureNum];
    if (augmented != nullptr){
        for(size_t i = 0;i < featureNum;i++)
            delete[] augmented[i];
        delete[] augmented;
    }
    return;
}
void NonNaiveBayesClassifier::Train(const std::vector<Sample>& samples,const float* classProbs){
    featureNum = samples[0].getFeatures().size(); //select all
    para.clear();
    para.reserve(Classes::counter);
    unsigned int classNum = Classes::counter;
    std::vector<double> classifiedFeaturesAvg[classNum],classifiedFeaturesVar[classNum];
    for (int i = 0; i < classNum; i++){
        classifiedFeaturesAvg[i].assign(featureNum,0.0);
        classifiedFeaturesVar[i].assign(featureNum,0.0);
    }
    std::vector<size_t> classRecordNum(classNum,0);
    std::vector<std::vector<vFloat>> sampleBucket(classNum);
    for (unsigned int i = 0; i < classNum; i++)
        sampleBucket[i].reserve(featureNum);
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
    for (std::vector<Sample>::const_iterator it = samples.begin(); it != samples.end(); it++){
        unsigned int label = static_cast<unsigned int>(it->getLabel());
        const vFloat& sampleFeature = it->getFeatures();
        for (unsigned int i = 0; i < featureNum; i++)
            classifiedFeaturesVar[label][i] += (sampleFeature[i] - classifiedFeaturesAvg[label][i]) * (sampleFeature[i] - classifiedFeaturesAvg[label][i]);
    }
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < featureNum; j++)
            classifiedFeaturesVar[i][j] = std::sqrt(classifiedFeaturesVar[i][j]/classRecordNum[i]);
    return;
};
void NonNaiveBayesClassifier::LUdecomposition(float** matrix, float** L, float** U){
    for (int i = 0; i < featureNum; ++i) { // init LU
        for (int j = 0; j < featureNum; ++j) {
            L[i][j] = 0;
            U[i][j] = matrix[i][j];
        }
        L[i][i] = 1;
    }
    for (int i = 0; i < featureNum; ++i) { // LU decomposition
        for (int j = i; j < featureNum; ++j) 
            for (int k = 0; k < i; ++k) 
                U[i][j] -= L[i][k] * U[k][j];
        for (int j = i + 1; j < featureNum; ++j) {
            for (int k = 0; k < i; ++k)
                L[j][i] -= L[j][k] * U[k][i];
            L[j][i] = U[j][i] / U[i][i];
        }
    }
}
double NonNaiveBayesClassifier::determinant(float** matrix) {
    float** L = new float*[featureNum];
    float** U = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++){
        L[i] = new float[featureNum];
        U[i] = new float[featureNum];
    }
    LUdecomposition(matrix, L, U);
    double det = 1.0;
    for (int i = 0; i < featureNum; ++i)
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