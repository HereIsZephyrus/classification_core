#ifndef FRUITHPP
#define FRUITHPP
#include <opencv2/opencv.hpp>
#include <iostream>
#include "func.hpp"
#include "process.hpp"

int FruitMain();
void PreProcess(const cv::Mat& rawImage, cv::Mat&processed);
int HistMethod(const cv::Mat& rawImage);
int BayersMethod(const cv::Mat& correctImage);
int FisherMethod(const cv::Mat& correctImage);
#endif