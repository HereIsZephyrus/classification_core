#ifndef LANDUSEHPP
#define LANDUSEHPP
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
#include "rs_classifier.hpp"
int LanduseMain();
int SeriesMain();
namespace weilaicheng{
bool StudySamples(land_StaticPara* classParas,std::vector<land_Sample>& dataset);
}
namespace ningbo{
using std::vector;
bool StudySamples(urban_StaticPara* classParas,std::vector<urban_Sample>& dataset);
std::shared_ptr<Classified> classifySingleYear(const vector<urban_Sample>& dataset,int year,const vector<std::string>& classifierForUse);
bool SeriesAnalysis(const vector<std::unique_ptr<Classified>>& imageSeries,vector<double>& increasingRate,vector<char>& increasingDirection);
bool FindBestClassifier(std::shared_ptr<Classified> classified,const vector<std::unique_ptr<T_Classifier<UrbanChange>>>& classifiers,const vector<cv::Mat>& classifiedImages);
bool ReadTrueClasses(vector<UrbanChange>& trueClasses2022);
bool ReadRawImage(int year,cv::Mat& rawImage,vFloat& MINVAL,vFloat& MAXVAL);
}
#endif