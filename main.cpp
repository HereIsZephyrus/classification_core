//#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "fruit/fruit.hpp"

int LanduseMain();
int SeriesMain();
int main(int argc, char* argv[]) {
    std::string programType = "fruit";
    //std::cin>>programType;
    if (programType == "fruit"){
        FruitMain();
    }else if (programType == "landuse"){
        LanduseMain();
    }else if (programType == "series"){
        SeriesMain();
    }
    return 0;
}
int LanduseMain(){
    return 0;
}
int SeriesMain(){
    return 0;
}