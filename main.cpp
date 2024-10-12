//#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "func.hpp"
#include "process.hpp"
int main() {
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
    cv::imshow("correct Image", correctImage);
    cv::waitKey(0);
    cv::destroyWindow("correct Image");
    
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