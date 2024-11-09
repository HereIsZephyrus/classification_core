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
#endif