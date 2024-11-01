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
    SpectralType
};
typedef std::vector<LandCover> vCovers;
typedef T_StaticPara<LandCover> land_StaticPara;
typedef T_Sample<LandCover> land_Sample;
typedef T_NaiveBayesClassifier<LandCover> land_NaiveBayesClassifier;
typedef T_NonNaiveBayesClassifier<LandCover> land_NonNaiveBayesClassifier;
typedef T_FisherClassifier<LandCover> land_FisherClassifier;
}
#endif