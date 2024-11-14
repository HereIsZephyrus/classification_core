#ifndef RSCLASSIFIERHPP
#define RSCLASSIFIERHPP
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
#include <map>
#include "../func.hpp"
#include "../t_classifier.hpp"
#include "image.hpp"
namespace weilaicheng{
constexpr float trainRatio = 0.8f;
enum LandCover : unsigned int{
    Water,
    Greenland,
    Bareland,
    Imprevious,
    CoverType,
    Edge,
    UNCLASSIFIED
};
enum Spectra : unsigned int{
    Blue,
    Green,
    Red,
    NIR,
    FalseHue,
    SpectralNum
};
typedef std::vector<LandCover> vCovers;
extern std::unordered_map<LandCover,std::string> classFolderNames;
extern std::unordered_map<LandCover,cv::Scalar> classifyColor;
extern vFloat MAXVAL,MINVAL;
}
typedef T_StaticPara<weilaicheng::LandCover> land_StaticPara;
typedef T_Sample<weilaicheng::LandCover> land_Sample;
namespace weilaicheng{
class land_NaiveBayesClassifier : public T_NaiveBayesClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
    bool CalcClassProb(float* prob) override;
};
class land_FisherClassifier : public T_FisherClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
};
class land_SVMClassifier : public T_SVMClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
public:
    void Train(const std::vector<land_Sample>& dataset) override;
};
class land_BPClassifier : public T_BPClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
};
class land_RandomForestClassifier : public T_RandomForestClassifier<weilaicheng::LandCover>{
    size_t getClassNum() const override{return weilaicheng::LandCover::CoverType;}
};
bool GenerateClassifiedImage(const cv::Mat& rawimage,cv::Mat& classified,const std::vector<std::vector<LandCover>>& pixelClasses);
}
#endif