#include <iostream>
#include "fruit/fruit.hpp"
#include "landuse/landuse.hpp"

int main(int argc, char* argv[]) {
    std::string programType = "series";
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