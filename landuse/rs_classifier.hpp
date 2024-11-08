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
template <typename classType>
class assess_Classifier{
protected:
    float precision,recall,f1;
    void Examine(const std::vector<T_Sample<classType>>& samples){
        size_t pcorrectNum = 0, rcorrectNum = 0;
        for (std::vector<T_Sample<classType>>::const_iterator it = samples.begin(); it != samples.end(); it++){
            if (it->isTrainSample())
                continue;
            if (Predict(it->getFeatures()) == it->getLabel())
                pcorrectNum++;
            if (it->getLabel() == Predict(it->getFeatures()))
                rcorrectNum++;
        }
        precision = static_cast<float>(pcorrectNum)/samples.size();
        recall = static_cast<float>(rcorrectNum)/samples.size();
        f1 = 2*precision*recall/(precision+recall);
    }
public:
    virtual classType Predict(const vFloat& x) = 0;
    virtual void Classify(const cv::Mat& rawImage) = 0;
    void Print(){
        std::cout << "Precision: " << precision << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "F1: " << f1 << std::endl;
    }
};
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
typedef T_StaticPara<LandCover> land_StaticPara;
typedef T_tagSample<LandCover> land_Sample;
using namespace bayes;
using namespace linear;
class land_NaiveBayesClassifier : public T_NaiveBayesClassifier<LandCover>, public assess_Classifier<LandCover>{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_FisherClassifier : public T_FisherClassifier<LandCover>, public assess_Classifier<LandCover>{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_SVMClassifier : public T_SVMClassifier<LandCover>, public assess_Classifier<LandCover>{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_BPClassifier : public T_BPClassifier<LandCover>, public assess_Classifier<LandCover>{
public:
    void Classify(const cv::Mat& rawImage);
};
class land_RandomClassifier : public T_RandomForestClassifier<LandCover>, public assess_Classifier<LandCover>{
public:
    void Classify(const cv::Mat& rawImage);
};
#endif