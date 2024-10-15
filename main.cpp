//#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "func.hpp"
#include "process.hpp"

void PreProcess(const cv::Mat& rawImage, cv::Mat&processed){
    cv::Mat correctImage;
    CorrectImage(rawImage,correctImage);
    processed = correctImage.clone();
    FetchShadow(correctImage,processed);
    return;
}
int HistMethod(const cv::Mat& rawImage){
    using namespace hist;
    cv::Mat correctImage;
    CorrectImage(rawImage,correctImage);
    cv::Mat flatImage = correctImage.clone();
    FetchShadow(correctImage,flatImage);
    SmoothImage(correctImage,flatImage);
    //cv::Mat flatImage = cv::imread("flatImage.png");
    cv::Mat totalMask;
    CreateMask(flatImage,totalMask);
    cv::Mat firstConvexImage,secondConvexImage;
    FirstConvexHull(totalMask,firstConvexImage);
    SecondConvexHull(firstConvexImage,totalMask,secondConvexImage);
    cv::Mat blendedImage = cv::max(rawImage, secondConvexImage);
    cv::imshow("Blended Image", blendedImage);
    cv::waitKey(0);
    cv::destroyWindow("Blended Image");
    ClassifityFruits(rawImage,correctImage,flatImage,secondConvexImage);
    return 0;
}
int BayersMethod(const cv::Mat& correctImage){
    using namespace bayes;
    unsigned int classesNum = Classes::counter;
    float* classProbs = new float[classesNum];
    CalcClassProb(classProbs);
    StaticPara* classParas = new StaticPara[classesNum];
    for (unsigned int classID = 0; classID < classesNum; classID++)
        classParas[classID].InitClassType(static_cast<Classes>(classID));
    std::vector<Sample> dataset;
    StudySamples(classParas,dataset);
    delete[] classParas;
    NaiveBayesClassifier* classifier = new NaiveBayesClassifier();
    //NonNaiveBayesClassifier* classifier = new NonNaiveBayesClassifier();
    classifier->Train(dataset,classProbs);
    delete[] classProbs;
    ClassMat patchClasses,pixelClasses;
    BayesClassify(correctImage,classifier,patchClasses);
    cv::Mat classified;
    DownSampling(patchClasses,pixelClasses);
    GenerateClassifiedImage(correctImage,classified,pixelClasses);
    cv::imshow("classified Image", classified);
    cv::waitKey(0);
    cv::destroyWindow("classified Image");
    cv::imwrite(classifier->printPhoto(), classified);
    return 0;
}
int FisherMethod(){
   return 0; 
}
int main() {
    std::cout << "Which method do you want to use?" << std::endl;
    std::cout << "1. Histo Method" << std::endl;
    std::cout << "2. bayes Method" << std::endl;
    std::cout << "3. Fisher Method" << std::endl;
    int type = 2;
    //std::cin>>type;
    cv::Mat rawImage = cv::imread("../images/spot.jpeg");
    if (rawImage.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }
    cv::Mat processed;
    cv::imshow("raw Image", rawImage);
    cv::waitKey(0);
    cv::destroyWindow("raw Image");
    switch (type){
        case 1:
            HistMethod(rawImage);
            break;
        case 2:{
            PreProcess(rawImage,processed);
            BayersMethod(processed);
            break;
        }
        case 3:
            break;
        default:
            std::cout << "Wrong input" << std::endl;
            break;
    }
    return 0;
}