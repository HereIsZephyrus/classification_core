#include <algorithm>
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
    GenerateFeatureImage(rawImage);
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
    vector<std::string> classifierForUse = {"bayes","fisher","svm","bp","rf"};
    vector<YearImage> classifyYears;
    //vector<int> classifyYears = {2022}; // TestClassifierFor2022
    // supervision train sample
    unsigned int classNum = LandCover::CoverType;
    urban_StaticPara* classParas = new urban_StaticPara[classNum];
    for (unsigned int classID = 0; classID < classNum; classID++)
        classParas[classID].InitClassType(static_cast<LandCover>(classID));
    vector<urban_Sample> dataset;
    StudySamples(classParas,dataset);
    delete[] classParas;
    for (int year = 1997; year <=2022; year+=5){
        cv::Mat featureImage;
        GenerateFeatureImage(year,featureImage,MINVAL,MAXVAL);
        classifyYears.push_back(std::make_pair(year,featureImage));
    }
    vector<vector<LandCover>> trueClasses2022;
    ReadTrueClasses(trueClasses2022);

    // classifiy series
    vector<std::shared_ptr<Classified>> imageSeries;
    for (vector<YearImage>::iterator it = classifyYears.begin(); it != classifyYears.end(); it++)
        imageSeries.push_back(ClassifySingleYear(dataset,*it,classifierForUse));
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
}//namespace weilaicheng

namespace ningbo{
using std::vector;
using vClasses = vector<LandCover>;
using classMat = vector<vector<LandCover>>;
std::shared_ptr<Classified> ClassifySingleYear( const vector<urban_Sample>& dataset,
                                                const YearImage& yearImage,
                                                const vector<std::string>& classifierForUse){
    using BaseClassifier = T_Classifier<LandCover>;
    std::shared_ptr<Classified> classified = nullptr;
    vector<std::unique_ptr<BaseClassifier>> classifiers;
    vector<classMat> pixelClasses;
    for (vector<std::string>::const_iterator classifierName = classifierForUse.begin(); classifierName != classifierForUse.end(); classifierName++){
        BaseClassifier* classifier;
        if (*classifierName == "bayes"){
            classifier = new urban_NaiveBayesClassifier();
            classifier->setYear(yearImage.first);
        }
        else if (*classifierName == "fisher")
            classifier = new urban_FisherClassifier();
        else if (*classifierName == "svm")
            classifier = new urban_SVMClassifier();
        else if (*classifierName == "bp")
            classifier = new urban_BPClassifier();
        else if (*classifierName == "rf")
            classifier = new urban_RandomForestClassifier();
        classifier->Train(dataset);
        classMat singleYearPixelClasses;
        classifier->Classify(yearImage.second,singleYearPixelClasses,LandCover::Edge,MINVAL,MAXVAL,classifierKernelSize);
        pixelClasses.push_back(singleYearPixelClasses);
        classifier->Examine(dataset);
        classifiers.push_back(std::unique_ptr<BaseClassifier>(classifier));
    }
    CombinedClassifier(classified,classifiers,pixelClasses,yearImage.second,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
    classified->CalcUrbanMorphology(classifyColor[LandCover::Imprevious]);
    classified->Examine(dataset);
    return classified;
}
bool SeriesAnalysis(const vector<std::shared_ptr<Classified>>& imageSeries,
                    vector<double>& increasingRate,
                    vector<char>& increasingDirection){
    increasingRate.clear();
    increasingDirection.clear();
    vector<std::shared_ptr<Classified>>::const_iterator image = imageSeries.begin();
    double lastArea = (*image)->getArea();
    std::shared_ptr<cv::Mat> lastImage = (*image)->getUrbanMask();
    ++image;
    for (;image != imageSeries.end(); image++){
        double currentArea = (*image)->getArea();
        std::shared_ptr<cv::Mat> currentImage = (*image)->getUrbanMask();
        increasingRate.push_back((currentArea - lastArea) / lastArea);
        increasingDirection.push_back(UrbanMaskAnalysis(lastImage,currentImage));
        lastArea = currentArea;
        lastImage = currentImage;
    }
    return true;
}
bool ReadTrueClasses(classMat& trueClasses2022){
    using namespace ningbo;
    cv::Mat classifiedImage = cv::imread("../landuse/ningbo/true_classified-clip.tif");
    for (int i = 0; i < classifiedImage.rows; i++){
        vClasses trueClassesRow;
        for (int j = 0; j < classifiedImage.cols; j++){
            uchar value = classifiedImage.at<uchar>(j,i);
            if (value == 10 || value == 30)
                trueClassesRow.push_back(LandCover::Greenland);
            else if (value == 40)
                trueClassesRow.push_back(LandCover::CropLand);
            else if (value == 50)
                trueClassesRow.push_back(LandCover::Imprevious);
            else if (value == 60)
                trueClassesRow.push_back(LandCover::Bareland);
        }
        trueClasses2022.push_back(trueClassesRow);
    }
    return true;
}
bool CombinedClassifier(std::shared_ptr<Classified> classified,
                        const vector<std::unique_ptr<T_Classifier<LandCover>>>& classifiers,
                        const vector<vector<vector<LandCover>>> pixelClasses,
                        const cv::Mat& featureImage,
                        const vFloat& minVal,const vFloat& maxVal,int classifierKernelSize,
                        const std::unordered_map<LandCover,cv::Scalar>& classifyColor){
    using ClassMat = vector<vector<LandCover>>;
    vector<float> accuracy;
    for (vector<std::unique_ptr<T_Classifier<LandCover>>>::const_iterator classifier = classifiers.begin(); classifier != classifiers.end(); classifier++)
        accuracy.push_back((*classifier)->accuracy.getComprehensiveAccuracy());
    ClassMat combinedClasses;
    size_t classifierNum = classifiers.size(),classesRow = pixelClasses.size(),classesCol = pixelClasses[0].size();
    for (size_t y = 0; y < classesRow; y++){
        vector<LandCover> combinedClassesRow;
        for (size_t x = 0; x < classesCol; x++){
            std::unordered_map<LandCover,float> voteRes;
            for (size_t i = 0 ; i < classifierNum; i++)
                voteRes[pixelClasses[i][y][x]] += accuracy[i];
            LandCover resClass = LandCover::UNCLASSIFIED;
            float maxProb = 0.0;
            for (std::unordered_map<LandCover,float>::const_iterator vote = voteRes.begin(); vote != voteRes.end(); vote++)
                if (vote->second > maxProb){
                    maxProb = vote->second;
                    resClass = vote->first;
                }
            combinedClassesRow.push_back(resClass);
        }
        combinedClasses.push_back(combinedClassesRow);
    }
    cv::Mat classifiedImage = cv::Mat::zeros(featureImage.rows, featureImage.cols, CV_8UC3);
    classifiedImage.setTo(cv::Scalar(255,255,255));
    int y = 0;
    for (typename ClassMat::const_iterator row = combinedClasses.begin(); row != combinedClasses.end(); row++,y+=classifierKernelSize/2){
        int x = 0;
        for (typename vClasses::const_iterator col = row->begin(); col != row->end(); col++,x+=classifierKernelSize/2){
            if (x >= featureImage.cols - classifierKernelSize/2)
                break;
            cv::Rect window(x,y,classifierKernelSize/2,classifierKernelSize/2);
            classifiedImage(window) = classifyColor.at(*col);
        }
        if (y >= featureImage.rows - classifierKernelSize/2)
            break;
    }
    classified->setImage(classifiedImage);
    return true;
}
bool StudySamples(urban_StaticPara* classParas,std::vector<urban_Sample>& dataset){
    namespace fs = std::filesystem;
    srand(time(0));
    cv::Mat rawImage;
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
            urban_Sample sample(static_cast<LandCover>(classID),data,isTrain);
            dataset.push_back(sample);
        }
    }
    for (std::vector<urban_Sample>::iterator data = dataset.begin(); data != dataset.end(); data++)
        data->scalingFeatures(MAXVAL,MINVAL);
    return true;
}
}//namespace ningbo