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
    generateFeatureImage(rawImage);
    land_StaticPara* classParas = new land_StaticPara[classesNum];
    for (unsigned int classID = 0; classID < classesNum; classID++)
        classParas[classID].InitClassType(static_cast<LandCover>(classID));
    std::vector<land_Sample> dataset;
    StudySamples(classParas,dataset);
    delete[] classParas;
    {
        land_NaiveBayesClassifier* bayes = new land_NaiveBayesClassifier();
        bayes->Train(dataset);
        cv::Mat classified;
        bayes->Classify(rawImage,classified,LandCover::Edge,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
        cv::imshow("Naive Bayes", classified);
        bayes->Examine(dataset);
        bayes->Print(classified,classFolderNames);
        cv::waitKey(0);
        cv::destroyWindow("Naive Bayes");
    }
    {
        land_FisherClassifier* fisher = new land_FisherClassifier();
        fisher->Train(dataset);
        cv::Mat classified;
        fisher->Classify(rawImage,classified,LandCover::Edge,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
        cv::imshow("Fisher", classified);
        fisher->Examine(dataset);
        fisher->Print(classified,classFolderNames);
        cv::waitKey(0);
        cv::destroyWindow("Fisher");
    }
    {
        land_SVMClassifier* svm = new land_SVMClassifier();
        svm->Train(dataset);
        cv::Mat classified;
        svm->Classify(rawImage,classified,LandCover::Edge,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
        cv::imshow("SVM", classified);
        svm->Examine(dataset);
        svm->Print(classified,classFolderNames);
        cv::waitKey(0);
        cv::destroyWindow("SVM");
    }
    {
        land_BPClassifier* bp = new land_BPClassifier();
        bp->Train(dataset);
        cv::Mat classified;
        bp->Classify(rawImage,classified,LandCover::Edge,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
        cv::imshow("BP", classified);
        bp->Examine(dataset);
        bp->Print(classified,classFolderNames);
        cv::waitKey(0);
        cv::destroyWindow("BP");
    }
    {
        land_RandomForestClassifier* randomforest = new land_RandomForestClassifier();
        randomforest->Train(dataset);
        cv::Mat classified;
        randomforest->Classify(rawImage,classified,LandCover::Edge,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
        cv::imshow("Random Forest", classified);
        randomforest->Examine(dataset);
        randomforest->Print(classified,classFolderNames);
        cv::waitKey(0);
        cv::destroyWindow("Random Forest");
    }
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
        std::string classFolderPath = suitFolderPath + classFolderNames[static_cast<LandCover>(classID)] + "/";
        if (!fs::exists(classFolderPath))
            continue;
        for (const auto& entry : fs::recursive_directory_iterator(classFolderPath)) {
            const std::string suffix = ".tif";
            size_t pos = std::string(entry.path()).rfind(suffix);
            if (pos == std::string::npos || pos != std::string(entry.path()).length() - suffix.length())
                continue;
            classParas[classID].Sampling(entry.path());
        }
    }
    for (unsigned int i = 0; i < Spectra::SpectralNum * 2; i++){
        MAXVAL[i] = 0;
        MINVAL[i] = 1e6;
    }
    for (unsigned int classID = 0; classID < LandCover::CoverType; classID++){
        const std::vector<vFloat>& avg = classParas[classID].getAvg();
        const std::vector<vFloat>& var = classParas[classID].getVar();
        const unsigned int recordNum = classParas[classID].getRecordsNum();
        for (unsigned int i = 0; i < recordNum; i++){
            vFloat data;
            for (unsigned int d = 0; d < Spectra::SpectralNum; d++){
                MAXVAL[d * 2] = std::max(MAXVAL[d],avg[i][d]);
                MINVAL[d * 2] = std::min(MINVAL[d],avg[i][d]);
                MAXVAL[d * 2 + 1] = std::max(MAXVAL[d],var[i][d]);
                MINVAL[d * 2 + 1] = std::min(MINVAL[d],var[i][d]);
                data.push_back(avg[i][d]);
                data.push_back(var[i][d]);
            }
            //bool isTrain = (rand()%10) <= (trainRatio*10);
            bool isTrain = i <= (recordNum * trainRatio);
            land_Sample sample(static_cast<LandCover>(classID),data,isTrain);
            dataset.push_back(sample);
        }
    }
    for (std::vector<land_Sample>::iterator data = dataset.begin(); data != dataset.end(); data++)
        data->scalingFeatures(MAXVAL,MINVAL);
    return true;
}
}