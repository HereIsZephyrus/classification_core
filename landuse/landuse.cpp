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
        delete bayes;
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
        delete fisher;
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
        delete svm;
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
        delete bp;
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
        delete randomforest;
    }
    return 0;
}
int SeriesMain(){
    using namespace ningbo;
    using std::vector;
    // supervision train sample
    unsigned int classNum = UrbanChange::LandType;
    urban_StaticPara* = classParas = new urban_StaticPara[classNum];
    for (unsigned int classID = 0; classID < classNum; classID++)
        classParas[classID].InitClassType(static_cast<UrbanChange>(classID));
    vector<urban_Sample> dataset;
    StudySamples(classParas,dataset);
    delete[] classParas;
    vector<UrbanChange> trueClasses2022;
    ReadTrueClasses(trueClasses2022);

    // classifiy series
    vector<std::shared_ptr<Classified>> imageSeries;
    vector<std::string> classifierForUse = {"bayes","fisher","svm","bp","rf"};
    vector<int> classifyYears = {1997,2002,2007,2012,2017,2022};
    //vector<int> classifyYears = {2022}; // TestClassifierFor2022
    for (int year = 1997; year <=2022; year+=5)
        imageSeries.push_back(classifySingleYear(dataset,year,classifierForUse));
    vector<double> increasingRate;
    vector<char> increasingDirection;
    SeriesAnalysis(imageSeries,increasingRate,increasingDirection);
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

namespace ningbo{
using std::vector;
bool StudySamples(urban_StaticPara* classParas,vector<urban_Sample>& dataset){
    return true;
}
std::shared_ptr<Classified> classifySingleYear(const vector<urban_Sample>& dataset,int year,const vector<std::string>& classifierForUse){
    using BaseClassifier = T_Classifier<UrbanChange>;
    cv::Mat rawImage;
    ReadRawImage(year,rawImage,MINVAL,MAXVAL);
    std::shared_ptr<Classified> classified = nullptr;
    vector<std::unique_ptr<BaseClassifier>> classifiers;
    vector<cv::Mat> classifiedImages;
    for (vector<std::string>::const_iterator classifierName = classifierForUse.begin(); classifierName != classifierForUse.end(); classifierName++){
        BaseClassifier* classifier;
        if (*classifierName == "bayes")
            classifier = new urban_NaiveBayesClassifier();
        else if (*classifierName == "fisher")
            classifier = new urban_FisherClassifier();
        else if (*classifierName == "svm")
            classifier = new urban_SVMClassifier();
        else if (*classifierName == "bp")
            classifier = new urban_BPClassifier();
        else if (*classifierName == "rf")
            classifier = new urban_RandomForestClassifier();
        classifier->Train(dataset);
        cv::Mat classifiedImage;
        classifier->Classify(rawImage,classifiedImage,UrbanChange::Edge,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
        classifiedImages.push_back(classifiedImage);
        classifier->Examine(dataset);
        classifiers.push_back(std::unique_ptr<BaseClassifier>(classifier));
    }
    FindBestClassifier(classified,classifiers,classifiedImages);
    return classified;
}
bool SeriesAnalysis(const vector<std::unique_ptr<Classified>>& imageSeries,
                    vector<double>& increasingRate,vector<char>& increasingDirection){
    return true;
}
bool ReadRawImage(int year,cv::Mat& rawImage,vFloat& MINVAL,vFloat& MAXVAL){
    return true;
}
bool ReadTrueClasses(vector<UrbanChange>& trueClasses2022){
    return true;
}
bool FindBestClassifier(std::shared_ptr<Classified> classified,
                        const vector<std::unique_ptr<T_Classifier<UrbanChange>>>& classifiers,
                        const vector<cv::Mat>& classifiedImages){
    return true;
}