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
extern std::unordered_map<LandCover,cv::Vec3b> classifyColor;
extern vFloat MAXVAL,MINVAL;
constexpr int classifierKernelSize = 9;
constexpr float trainRatio = 0.8f;
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
public:
    land_RandomForestClassifier():T_RandomForestClassifier<LandCover>(10,10,2,1,200){}
};
}

namespace ningbo{
enum LandCover : unsigned int{
    Water,
    Greenland,
    Bareland,
    CropLand,
    Imprevious,
    CoverType,
    Cloud,
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
extern std::unordered_map<LandCover,std::string> classFolderNames;
extern std::unordered_map<LandCover,cv::Vec3b> classifyColor;
extern vFloat MINVAL,MAXVAL;
constexpr int classifierKernelSize = 5;
constexpr float trainRatio = 0.8f;
}
typedef T_StaticPara<ningbo::LandCover> urban_StaticPara;
typedef T_Sample<ningbo::LandCover> urban_Sample;
typedef std::pair<int,cv::Mat> YearImage;
namespace ningbo{
class urban_NaiveBayesClassifier : public T_NaiveBayesClassifier<LandCover>{
    int year;
    size_t getClassNum() const override{return LandCover::CoverType;}
    bool CalcClassProb(float* prob) override;
public:
    urban_NaiveBayesClassifier(){}
    int getYear() const {return year;}
    void setYear(int year) override{this->year = year;}
};
class urban_FisherClassifier : public T_FisherClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
};
class urban_SVMClassifier : public T_SVMClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
//public:
    //void Train(const std::vector<urban_Sample>& dataset) override;
};
class urban_BPClassifier : public T_BPClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
};
class urban_RandomForestClassifier : public T_RandomForestClassifier<LandCover>{
    size_t getClassNum() const override{return LandCover::CoverType;}
public:
    urban_RandomForestClassifier():T_RandomForestClassifier<LandCover>(50,40,2,1,250){}
};
class Classified{
    cv::Mat urbanMask;
    double area;
public:
    cv::Mat image;
    Accuracy<LandCover> accuracy;
    Classified() = default;
    const cv::Mat& getUrbanMask() const {return urbanMask;}
    double getArea() const {return area;}
    void setImage(const cv::Mat& classifiedImage){image = classifiedImage.clone();}
    void CalcUrbanMorphology(const vector<cv::Vec3b>& impreviousColor);
    void Examine(const vector<urban_Sample>& samples);
    void Print();
};
}
#endif