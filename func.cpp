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
vFloat CalcConv(const std::vector<vFloat>& x, vFloat avgx, const std::vector<vFloat>& y, vFloat avgy){
    vFloat res;
    const size_t n = x.size();
    for (unsigned int d = 0; d < Demisions::dim; d++){
        double tempRes = 0.0f;
        for (size_t i = 0; i<n; i++)
            tempRes += (x[i][d] - avgx[d]) * (y[i][d] - avgy[d]);
        tempRes /= (n-1);
        res.push_back(static_cast<float>(tempRes));
    }
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
    //std::cout<<featureNum<<std::endl;
    for (unsigned int d = 0; d < featureNum; d++){
        float pd = x[d] - para[static_cast<Classes>(classID)].mu[d];
        float vars = para[static_cast<Classes>(classID)].sigma[d] * para[static_cast<Classes>(classID)].sigma[d];
        double exponent = std::exp(static_cast<double>(- pd * pd / (2 * vars)));
        double normalize = 1.0f / (sqrt(2 * CV_PI) * vars);
        res *= normalize * exponent;
    }
    return res;
}
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
/*
double NonNaiveBayesClassifier::CalculateClassProbability(unsigned int classID,const vFloat& x){
    return 0.0f;
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
void NonNaiveBayesClassifier::Train(const std::vector<Sample>& samples,const float* classProbs){
    const unsigned int classNum = Classes::counter;
    convMat = new vFloat*[classNum];
    invConvMat = new vFloat*[classNum];
    for (unsigned int classID = 0; classID < classNum; classID++){
        convMat[classID] = new vFloat[classNum];
        invConvMat[classID] = new vFloat[classNum];
    }
    for (unsigned int classID = 0; classID < classNum; classID++){
        para[static_cast<Classes>(classID)].mu = densityParas[classID].CombineMu(0,densityParas[classID].getRecordsNum());
        para[static_cast<Classes>(classID)].sigma = densityParas[classID].CombineSigma(0,densityParas[classID].getRecordsNum());
        para[static_cast<Classes>(classID)].w = classProbs[classID];
    }
    for (unsigned int classX = 0; classX < classNum; classX++){
        convMat[classX][classX] = para[static_cast<Classes>(classX)].sigma;
        for (unsigned int classY = classX + 1; classY < classNum; classY++){
            vFloat conv = CalcConv(densityParas[classX].getMu(),para[static_cast<Classes>(classX)].mu,densityParas[classY].getMu(),para[static_cast<Classes>(classY)].mu);
            convMat[classX][classY] = conv;
            convMat[classY][classX] = conv;
        }
    }

    return;
};
*/
};//namespace bayes