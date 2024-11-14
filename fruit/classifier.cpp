#include "classifier.hpp"
#include <fstream>
#include <filesystem>
using namespace Eigen;
namespace fruit{
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
}
using namespace fruit;
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
    using namespace fruit;
    cv::Mat patch = cv::imread(entryPath);
    if (patch.empty()){
        std::cerr << "Image not found!" << std::endl;
        return;
    }
    std::vector<cv::Mat> channels;
    GenerateFeatureChannels(patch,channels);
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
bool NaiveBayesClassifier::CalcClassProb(float* prob){
    unsigned int* countings = new unsigned int[Classes::counter];
    unsigned int totalRecord = 0;
    for (int i = 0; i < Classes::counter; i++)
        countings[i] = 0;
    std::string filename = "../fruit/sampling/suit3/classification.csv";
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "can't open file!" << filename << std::endl;
        return false;
    }
    std::getline(file, line);// throw header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;
        while (std::getline(ss, value, ',')) {}
        totalRecord++;
        value.pop_back();
        if (value == "Desk")
            countings[Classes::Desk]++;
        else if (value == "Apple")
            countings[Classes::Apple]++;
        else if (value == "Blackplum")
            countings[Classes::Blackplum]++;
        else if (value == "Dongzao")
            countings[Classes::Dongzao]++;
        else if (value == "Grape")
            countings[Classes::Grape]++;
        else if (value == "Peach")
            countings[Classes::Peach]++;
        else if (value == "Yellowpeach")
            countings[Classes::Yellowpeach]++;
    }
    file.close();
    for (int i = 0; i < Classes::counter; i++)
        prob[i] = static_cast<float>(countings[i]) / totalRecord;
    delete[] countings;
    return true;
}
bool NonNaiveBayesClassifier::CalcClassProb(float* prob){
    unsigned int* countings = new unsigned int[Classes::counter];
    unsigned int totalRecord = 0;
    for (int i = 0; i < Classes::counter; i++)
        countings[i] = 0;
    std::string filename = "../fruit/sampling/suit3/classification.csv";
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "can't open file!" << filename << std::endl;
        return false;
    }
    std::getline(file, line);// throw header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;
        while (std::getline(ss, value, ',')) {}
        totalRecord++;
        value.pop_back();
        if (value == "Desk")
            countings[Classes::Desk]++;
        else if (value == "Apple")
            countings[Classes::Apple]++;
        else if (value == "Blackplum")
            countings[Classes::Blackplum]++;
        else if (value == "Dongzao")
            countings[Classes::Dongzao]++;
        else if (value == "Grape")
            countings[Classes::Grape]++;
        else if (value == "Peach")
            countings[Classes::Peach]++;
        else if (value == "Yellowpeach")
            countings[Classes::Yellowpeach]++;
    }
    file.close();
    for (int i = 0; i < Classes::counter; i++)
        prob[i] = static_cast<float>(countings[i]) / totalRecord;
    delete[] countings;
    return true;
}
void NonNaiveBayesClassifier::train(const std::vector<Sample>& samples,const float* classProbs){
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
bool CalcChannelMeanStds(const vector<cv::Mat> & channels, vFloat & data){
    data.clear();
    for (vector<cv::Mat>::const_iterator it = channels.begin(); it != channels.end(); it++){
        cv::Scalar mean, stddev;
        cv::meanStdDev(*it, mean, stddev);
        data.push_back(cv::mean(*it)[0]);
        data.push_back(stddev[0] * stddev[0]);
    }
    return true;
}
bool LinearClassify(const cv::Mat& rawimage,FisherClassifier* classifer,std::vector<std::vector<Classes>>& patchClasses){
    using namespace fruit;
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
            fruit::GenerateFeatureChannels(sample, channels);
            CalcChannelMeanStds(channels, data);
            rowClasses.push_back(classifer->Predict(data));
        }
        patchClasses.push_back(rowClasses);
    }
    return true;
}