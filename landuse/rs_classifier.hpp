#ifndef RSCLASSIFIERHPP
#define RSCLASSIFIERHPP
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
#include <map>
#include <Eigen/Dense>
#include "../func.hpp"
#include "../t_classifier.hpp"
namespace weilaicheng{
enum LandCover : unsigned int{
    Water,
    Greenland,
    Bareland,
    Imprevious,
    CoverType,
    UNCLASSIFIED
};
enum Spectra : unsigned int{
    Blue,
    Green,
    Red,
    NIR,
    TrueSat,
    TrueHue,
    FalseSat,
    FalseHue,
    SpectralNum
};
typedef std::vector<LandCover> vCovers;
extern std::string classFolderNames[LandCover::CoverType];
extern std::unordered_map<LandCover,cv::Scalar> classifyColor;
}
typedef T_StaticPara<weilaicheng::LandCover> land_StaticPara;
typedef T_Sample<weilaicheng::LandCover> land_Sample;
namespace weilaicheng{
using namespace bayes;
using namespace linear;
class land_NaiveBayesClassifier : public T_NaiveBayesClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
    void Train(const std::vector<land_Sample>& dataset,const float* classProbs) override;
    bool CalcClassProb(float* prob);
public:
    void Classify(const cv::Mat& rawImage);
    double CalculateClassProbability(unsigned int classID,const vFloat& x) override;
    void Train(const std::vector<land_Sample>& dataset);
};
class land_FisherClassifier : public T_FisherClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
public:
    void Classify(const cv::Mat& rawImage);
    void Train(const std::vector<land_Sample>& dataset) override;
};
class land_SVMClassifier : public T_SVMClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
public:
    void Classify(const cv::Mat& rawImage);
    void Train(const std::vector<land_Sample>& dataset) override;
};
class land_BPClassifier : public T_BPClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
public:
    void Classify(const cv::Mat& rawImage);
    void Train(const std::vector<land_Sample>& dataset) override;
};
class land_RandomClassifier : public T_RandomForestClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
public:
    void Classify(const cv::Mat& rawImage);
    void Train(const std::vector<land_Sample>& dataset) override;
};
}
#endif