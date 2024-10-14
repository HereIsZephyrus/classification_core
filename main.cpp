//#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "func.hpp"
#include "process.hpp"

int HistMethod(){
    using namespace hist;
    cv::Mat rawimage = cv::imread("../images/spot.jpeg");
    if (rawimage.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }
    cv::imshow("raw Image", rawimage);
    cv::waitKey(0);
    cv::destroyWindow("raw Image");
    cv::Mat correctImage;
    CorrectImage(rawimage,correctImage);
    cv::Mat flatImage = correctImage.clone();
    fetchShadow(correctImage,flatImage);
    SmoothImage(correctImage,flatImage);
    //cv::Mat flatImage = cv::imread("flatImage.png");
    cv::Mat totalMask;
    CreateMask(flatImage,totalMask);
    cv::Mat firstConvexImage,secondConvexImage;
    FirstConvexHull(totalMask,firstConvexImage);
    SecondConvexHull(firstConvexImage,totalMask,secondConvexImage);
    cv::Mat blendedImage = cv::max(rawimage, secondConvexImage);
    cv::imshow("Blended Image", blendedImage);
    cv::waitKey(0);
    cv::destroyWindow("Blended Image");
    ClassifityFruits(rawimage,correctImage,flatImage,secondConvexImage);
    return 0;
}
int BayersMethod(){
    using namespace bayers;
    unsigned int classesNum = Classes::counter;
    float* probClass = new float[classesNum];
    CalcClassProb(probClass);
    StaticPara* classParas = new StaticPara[classesNum];
    for (unsigned int classID = 0; classID < classesNum; classID++)
        classParas[classID].InitClassType(static_cast<Classes>(classID));
    StudySamples(classParas);
    
    delete[] probClass;
    delete[] classParas;
    return 0;
}
int FisherMethod(){
   return 0; 
}
int main() {
    std::cout << "Which method do you want to use?" << std::endl;
    std::cout << "1. Histo Method" << std::endl;
    std::cout << "2. Bayers Method" << std::endl;
    std::cout << "3. Fisher Method" << std::endl;
    int type = 2;
    //std::cin>>type;
    switch (type){
        case 1:
            HistMethod();
            break;
        case 2:
            BayersMethod();
            break;
        case 3:
            break;
        default:
            std::cout << "Wrong input" << std::endl;
            break;
    }
    return 0;
}