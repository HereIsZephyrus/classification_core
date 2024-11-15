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
extern std::unordered_map<LandCover,std::string> classFolderNames;
extern std::unordered_map<LandCover,cv::Scalar> classifyColor;
extern vFloat MAXVAL,MINVAL;
constexpr int classifierKernelSize = 9;
}
typedef T_StaticPara<weilaicheng::LandCover> land_StaticPara;
typedef T_Sample<weilaicheng::LandCover> land_Sample;
namespace weilaicheng{
class land_NaiveBayesClassifier : public T_NaiveBayesClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
    bool CalcClassProb(float* prob) override;
};
class land_FisherClassifier : public T_FisherClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
};
class land_SVMClassifier : public T_SVMClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
public:
    void Train(const std::vector<land_Sample>& dataset) override;
};
class land_BPClassifier : public T_BPClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
};
class land_RandomForestClassifier : public T_RandomForestClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
};
}

namespace ningbo{
enum UrbanChange : unsigned int{
    Water,
    Greenland,
    Bareland,
    Imprevious,
    Cloud,
    LandType,
    Edge,
    UNCLASSIFIED
};
enum Spectra : unsigned int{
    Blue,
    Green,
    Red,
    NIR,
    SWIR1,
    SWIR2,
    SpectralNum
};
extern std::unordered_map<UrbanChange,std::string> classFolderNames;
extern std::unordered_map<UrbanChange,cv::Scalar> classifyColor;
extern vFloat MINVAL,MAXVAL;
constexpr int classifierKernelSize = 9;
}
typedef T_StaticPara<ningbo::UrbanChange> urban_StaticPara;
typedef T_Sample<ningbo::UrbanChange> urban_Sample;
namespace ningbo{
class urban_NaiveBayesClassifier : public T_NaiveBayesClassifier<UrbanChange>{
    int year;
    size_t getClassNum() const override{return UrbanChange::LandType;}
    bool CalcClassProb(float* prob) override;
public:
    urban_NaiveBayesClassifier(int year):year(year){}
    int getYear() const {return year;}
};
class urban_FisherClassifier : public T_FisherClassifier<UrbanChange>{
    size_t getClassNum() const override{return UrbanChange::LandType;}
};
class urban_SVMClassifier : public T_SVMClassifier<UrbanChange>{
    size_t getClassNum() const override{return UrbanChange::LandType;}
//public:
    //void Train(const std::vector<urban_Sample>& dataset) override;
};
class urban_BPClassifier : public T_BPClassifier<UrbanChange>{
    size_t getClassNum() const override{return UrbanChange::LandType;}
};
class urban_RandomForestClassifier : public T_RandomForestClassifier<UrbanChange>{
    size_t getClassNum() const override{return UrbanChange::LandType;}
};
class Classified{
    cv::Mat image,urbanMask;
    double area;
    void CalcUrbanMorphology();
public:
    Classified() = default;
};
}
#endif