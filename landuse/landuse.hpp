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
bool StudySamples(urban_StaticPara* classParas,vector<urban_Sample>& dataset);
std::shared_ptr<Classified> ClassifySingleYear( const vector<urban_Sample>& trainDataset,
                                                const vector<urban_Sample>& testDataset,
                                                const YearImage& yearImage,
                                                const vector<std::string>& classifierForUse);
bool SeriesAnalysis(const vector<std::shared_ptr<Classified>>& imageSeries,
                    vector<double>& increasingRate,vector<char>& increasingDirection);
bool CombinedClassifier(std::shared_ptr<Classified> classified,
                        const vector<std::unique_ptr<T_Classifier<LandCover>>>& classifiers,
                        const vector<vector<vector<LandCover>>> pixelClasses,
                        const cv::Mat& featureImage,
                        const vFloat& minVal,const vFloat& maxVal,int classifierKernelSize,
                        const std::unordered_map<LandCover,cv::Vec3b>& classifyColor);
bool ReadTrueClasses(vector<vector<LandCover>>& trueClasses2022);
bool StudyTrueClasses(const vector<vector<LandCover>>&trueClasses2022,const cv::Mat& RSimage,vector<urban_Sample>& trainDataset,size_t setSize);
}
#endif