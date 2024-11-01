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
    Wetland,
    Greenland,
    Bareland,
    Building,
    Road,
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
typedef T_StaticPara<LandCover> land_StaticPara;
typedef T_Sample<LandCover> land_Sample;
using namespace bayes;
using namespace linear;
class landuse_Classifier{
protected:
    float precision,recall,f1;
    void Examine();
public:
    virtual void Classify(const cv::Mat& rawImage) = 0;
    void Print();
};
class land_NaiveBayesClassifier : public T_NaiveBayesClassifier<LandCover>, public landuse_Classifier{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_FisherClassifier : public T_FisherClassifier<LandCover>, public landuse_Classifier{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_SVMClassifier : public T_SVMClassifier<LandCover>, public landuse_Classifier{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_FCNClassifier : public T_FCNClassifier<LandCover>, public landuse_Classifier{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_RandomClassifier : public T_RandomClassifier<LandCover>, public landuse_Classifier{
public:
    void Classify(const cv::Mat& rawImage);
};
#endif