#include <fstream>
#include <filesystem>
#include <ctime>
#include <cstdlib>
#include "landuse.hpp"
#include "image.hpp"
int LanduseMain(){
    using namespace weilaicheng;
    cv::Mat rawImage;
    unsigned int classesNum = LandCover::CoverType;
   //generateFeatureImage(rawImage);
    land_StaticPara* classParas = new land_StaticPara[classesNum];
    for (unsigned int classID = 0; classID < classesNum; classID++)
        classParas[classID].InitClassType(static_cast<LandCover>(classID));
    std::vector<land_Sample> dataset;
    StudySamples(classParas,dataset);
    delete[] classParas;
    land_NaiveBayesClassifier* bayes = new land_NaiveBayesClassifier();
    bayes->Train(dataset);
    bayes->Classify(rawImage);
    bayes->PrintPrecision();
    land_FisherClassifier* fisher = new land_FisherClassifier();
    fisher->Train(dataset);
    fisher->Classify(rawImage);
    fisher->PrintPrecision();
    land_SVMClassifier* svm = new land_SVMClassifier();
    svm->Train(dataset);
    svm->Classify(rawImage);
    svm->PrintPrecision();
    land_BPClassifier* bp = new land_BPClassifier();
    bp->Train(dataset);
    bp->Classify(rawImage);
    bp->PrintPrecision();
    land_RandomClassifier* randomforest = new land_RandomClassifier();
    randomforest->Train(dataset);
    randomforest->Classify(rawImage);
    randomforest->PrintPrecision();
    return 0;
}
int SeriesMain(){
    return 0;
}

namespace weilaicheng{
bool StudySamples(land_StaticPara* classParas,std::vector<land_Sample>& dataset){
    namespace fs = std::filesystem;
    srand(time(0));
    std::string suitFolderPath = "../landuse/sampling/";
    for (unsigned int classID = 0; classID < LandCover::CoverType; classID++){
        std::string classFolderPath = suitFolderPath + classFolderNames[classID] + "/";
        if (!fs::exists(classFolderPath))
            continue;
        for (const auto& entry : fs::recursive_directory_iterator(classFolderPath)) {
            classParas[classID].Sampling(entry.path());
        }
    }
    for (unsigned int classID = 0; classID < LandCover::CoverType; classID++){
        const std::vector<vFloat>& avg = classParas[classID].getAvg();
        const std::vector<vFloat>& var = classParas[classID].getVar();
        for (unsigned int i = 0; i < classParas[classID].getRecordsNum(); i++){
            vFloat data;
            for (unsigned int d = 0; d < Spectra::SpectralNum; d++){
                data.push_back(avg[i][d]);
                data.push_back(var[i][d]);
            }
            bool isTrain = (rand()%10) <= (trainRatio*10);
            land_Sample sample(static_cast<LandCover>(classID),data,isTrain);
            dataset.push_back(sample);
        }
    }
    return true;
}
}