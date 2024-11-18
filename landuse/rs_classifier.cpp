#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <memory>
#include <map>
#include <fstream>
#include <filesystem>
#include "rs_classifier.hpp"
using namespace Eigen;
template<>
void land_StaticPara::InitClassType(weilaicheng::LandCover ID){
    recordNum = 0;
    classID = ID;
    avg.clear();
    var.clear();
    return;
}
template<>
void land_StaticPara::Sampling(const std::string& entryPath){
    using weilaicheng::classifierKernelSize;
    using namespace weilaicheng;
    std::vector<cv::Mat> channels;
    cv::imreadmulti(entryPath,channels,cv::IMREAD_UNCHANGED);
    weilaicheng::GenerateFeatureChannels(channels);
    const unsigned int patchRows = channels[0].rows, patchCols = channels[0].cols;
    for (unsigned int left = 0; left < patchCols - classifierKernelSize + 1; left+=classifierKernelSize/2){
        for (unsigned int top = 0; top < patchRows - classifierKernelSize + 1; top+=classifierKernelSize/2){
            cv::Rect window(left, top, classifierKernelSize, classifierKernelSize);
            vFloat means, vars;
            for (unsigned int i = 0; i < Spectra::SpectralNum; i++){
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
namespace weilaicheng{
vFloat MAXVAL(Spectra::SpectralNum * 2),MINVAL(Spectra::SpectralNum * 2);
std::unordered_map<LandCover,std::string> classFolderNames = {
    {LandCover::Water,"water"},
    {LandCover::Greenland,"greenland"},
    {LandCover::Bareland,"bareland"},
    {LandCover::Imprevious,"imprevious"},
};
std::unordered_map<LandCover,cv::Scalar> classifyColor = {
    {LandCover::Water,cv::Scalar(255,0,0)}, // blue
    {LandCover::Imprevious,cv::Scalar(0,0,255)}, // red
    {LandCover::Bareland,cv::Scalar(42,42,165)}, // brzone,
    {LandCover::Greenland,cv::Scalar(0,255,0)}, // green
    {LandCover::Edge,cv::Scalar(255,255,255)}, // white
    {LandCover::UNCLASSIFIED,cv::Scalar(255,255,255)}, // black
};
bool land_NaiveBayesClassifier::CalcClassProb(float* prob){
    unsigned int* countings = new unsigned int[LandCover::CoverType];
    unsigned int totalRecord = 0;
    for (int i = 0; i < LandCover::CoverType; i++)
        countings[i] = 0;
    std::string filename = "../landuse/sampling/classification.csv";
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
        if (value == "Water")
            countings[LandCover::Water]++;
        else if (value == "Greenland")
            countings[LandCover::Greenland]++;
        else if (value == "Bareland")
            countings[LandCover::Bareland]++;
        else if (value == "Imprevious")
            countings[LandCover::Imprevious]++;
    }
    file.close();
    for (int i = 0; i < LandCover::CoverType; i++)
        prob[i] = static_cast<float>(countings[i]) / totalRecord;
    delete[] countings;
    return true;
}
void land_SVMClassifier::Train(const std::vector<land_Sample>& dataset){
    this->featureNum = dataset[0].getFeatures().size(); //select all
    unsigned int classNum = getClassNum();
    std::vector<int> greenlandPN,classCount[classNum];
    std::vector<int> greenlandLabel;
    bool selectTag = true;
    for (size_t i = 0; i < dataset.size(); i++){
        const T_Sample<LandCover>& sample = dataset[i];
        if (!sample.isTrainSample())
            continue;
        unsigned int classID = static_cast<unsigned int>(sample.getLabel());
        if (sample.getLabel() == LandCover::Greenland){
            greenlandPN.push_back(i);
            greenlandLabel.push_back(1);
        }else{
            if (selectTag){
                greenlandPN.push_back(i);
                greenlandLabel.push_back(-1);
            }
            selectTag = !selectTag;
        }
        classCount[classID].push_back(i);
    }
    std::unique_ptr<OVOSVM> greenlandClassifier = std::make_unique<OVOSVM>(LandCover::Greenland,LandCover::UNCLASSIFIED);
    greenlandClassifier->train(dataset,greenlandPN,greenlandLabel);
    classifiers.push_back(std::move(greenlandClassifier));
    unsigned int greenlandID = static_cast<unsigned int>(LandCover::Greenland);
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = i+1; j < classNum; j++){
            if (i == greenlandID || j == greenlandID)
                continue;
            std::vector<int> classPN = classCount[i];
            std::vector<int> classLabeli,classLabelj;
            classLabeli.assign(classCount[i].size(),1);
            classLabelj.assign(classCount[j].size(),-1);
            classLabeli.insert(classLabeli.end(), classLabelj.begin(), classLabelj.end());
            classPN.insert(classPN.end(), classCount[j].begin(), classCount[j].end());
            std::unique_ptr<OVOSVM> classifier = std::make_unique<OVOSVM>(static_cast<LandCover>(i),static_cast<LandCover>(j));
            classifier->train(dataset,classPN,classLabeli);
            classifiers.push_back(std::move(classifier));
        }
}
}

template<>
void urban_StaticPara::InitClassType(ningbo::LandCover ID){
    recordNum = 0;
    classID = ID;
    avg.clear();
    var.clear();
    return;
}
template<>
void urban_StaticPara::Sampling(const std::string& entryPath){
    using namespace ningbo;
    std::vector<cv::Mat> channels;
    cv::imreadmulti(entryPath,channels,cv::IMREAD_UNCHANGED);// !! not finished read multi tif and merge
    const unsigned int patchRows = channels[0].rows, patchCols = channels[0].cols;
    for (unsigned int left = 0; left < patchCols - classifierKernelSize + 1; left+=classifierKernelSize/2){
        for (unsigned int top = 0; top < patchRows - classifierKernelSize + 1; top+=classifierKernelSize/2){
            cv::Rect window(left, top, classifierKernelSize, classifierKernelSize);
            vFloat means, vars;
            for (unsigned int i = 0; i < Spectra::SpectralNum; i++){
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
namespace ningbo{
vFloat MAXVAL(Spectra::SpectralNum * 2),MINVAL(Spectra::SpectralNum * 2);
std::unordered_map<LandCover,std::string> classFolderNames = {
    {LandCover::Water,"water"},
    {LandCover::Greenland,"greenland"},
    {LandCover::Bareland,"bareland"},
    {LandCover::Imprevious,"imprevious"},
    {LandCover::CropLand,"cropLand"},
};
std::unordered_map<LandCover,cv::Scalar> classifyColor = {
    {LandCover::Water,cv::Scalar(255,0,0)}, // blue
    {LandCover::Imprevious,cv::Scalar(0,0,255)}, // red
    {LandCover::CropLand,cv::Scalar(0,255,255)}, // yellow
    {LandCover::Bareland,cv::Scalar(42,42,165)}, // brzone
    {LandCover::Greenland,cv::Scalar(0,255,0)}, // green
    {LandCover::Edge,cv::Scalar(255,255,255)}, // white
    {LandCover::Cloud,cv::Scalar(198,198,198)}, // gray
    {LandCover::UNCLASSIFIED,cv::Scalar(255,255,255)}, // black
};
bool urban_NaiveBayesClassifier::CalcClassProb(float* prob){
    unsigned int* countings = new unsigned int[LandCover::CoverType];
    unsigned int totalRecord = 0;
    for (int i = 0; i < LandCover::CoverType; i++)
        countings[i] = 0;
    std::string filename = "../landuse/ningbo/sampling/classification.csv";
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
        std::getline(ss, value, ',');
        if (std::to_string(year) != value)
            continue;
        while (std::getline(ss, value, ',')) {}
        totalRecord++;
        if (value == "Water")
            countings[LandCover::Water]++;
        else if (value == "Greenland")
            countings[LandCover::Greenland]++;
        else if (value == "Bareland")
            countings[LandCover::Bareland]++;
        else if (value == "Imprevious")
            countings[LandCover::Imprevious]++;
        else if (value == "Cropland")
            countings[LandCover::CropLand]++;
    }
    file.close();
    for (int i = 0; i < LandCover::CoverType; i++)
        prob[i] = static_cast<float>(countings[i]) / totalRecord;
    delete[] countings;
    return true;
}
void Classified::CalcUrbanMorphology(const cv::Scalar& impreviousColor){
    using cv::Point;
    vector<Point> impreviousPoints;
    int row = image.rows,col = image.cols;
    for (int y = 0; y < row; y++){
        for (int x = 0; x < col; x++){
            if (y == 3179 && x >= 4096)    break; 
            if (image.at<cv::Scalar>(y, x) == impreviousColor)
                impreviousPoints.push_back(Point(x, y));
            //I don't know why but it didn't work when the row reached the last row and the column reached 4096.
            //Luckily in this task, it won't affect the result.
        }
    }
    cv::Mat density = cv::Mat::zeros(image.size(), CV_32FC1);
    const double bandwidthSqr = 5.0 * 5.0;
    const double kernelScale = 1.0 / (2 * CV_PI * bandwidthSqr);
    for (vector<Point>::const_iterator point = impreviousPoints.begin(); point != impreviousPoints.end(); point++)
        for (int y = 0; y < row; y++)
            for (int x = 0; x < col; x++) {
                double distSqr = (point->x - x) * (point->x - x) + (point->y - y) * (point->y - y);
                if (distSqr < bandwidthSqr)
                    density.at<float>(y, x) += kernelScale * exp(-0.5 * distSqr / bandwidthSqr);
                if (y == 3179 && x >= 4096)    break; 
            }
    normalize(density, density, 0, 255, cv::NORM_MINMAX);
    density.convertTo(density, CV_8UC1);
    vector<float> kdeValues;
    density.reshape(1, density.total()).copyTo(kdeValues);
    std::sort(kdeValues.begin(), kdeValues.end());
    const float confidenceLevel = 0.95;
    int index = static_cast<int>(confidenceLevel * kdeValues.size());
    double threshold = kdeValues[index];
    cv::Mat maskImage = density >= threshold;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3, 3));
    cv::morphologyEx(maskImage, maskImage, cv::MORPH_OPEN, kernel);
    vector<vector<Point>> contours;
    vector<cv::Vec4i> hierarchy;
    findContours(maskImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat filledImage = cv::Mat::zeros(maskImage.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(filledImage, contours, (int)i, cv::Scalar(255), cv::FILLED);
    area = 0;
    for (int y = 0; y < row; y++)
        for (int x = 0; x < col; x++)
            if (filledImage.at<uchar>(y, x) == 255)
                ++area;
    urbanMask = filledImage.clone();
    return;
}
void Classified::Examine(const vector<urban_Sample>& samples){

}
void Classified::Print(){
    
}
}// namespace ningbo