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
        bayes->Classify(rawImage,classified,LandCover::Edge,LandCover::UNCLASSIFIED,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
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
        fisher->Classify(rawImage,classified,LandCover::Edge,LandCover::UNCLASSIFIED,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
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
        svm->Classify(rawImage,classified,LandCover::Edge,LandCover::UNCLASSIFIED,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
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
        bp->Classify(rawImage,classified,LandCover::Edge,LandCover::UNCLASSIFIED,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
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
        randomforest->Classify(rawImage,classified,LandCover::Edge,LandCover::UNCLASSIFIED,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
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
    //vector<std::string> classifierForUse = {"bayes","fisher","svm","bp","rf"};
    vector<std::string> classifierForUse = {"rf"};
    vector<YearImage> classifyYears;
    //vector<int> classifyYears = {2022}; // TestClassifierFor2022
    // supervision train sample
    unsigned int classNum = LandCover::CoverType;
    urban_StaticPara* classParas = new urban_StaticPara[classNum];
    for (unsigned int classID = 0; classID < classNum; classID++)
        classParas[classID].InitClassType(static_cast<LandCover>(classID));
    vector<urban_Sample> trainDataset;
    StudySamples(classParas,trainDataset);
    delete[] classParas;
    for (int year = 1997; year <=2022; year+=5){
        cv::Mat featureImage;
        GenerateFeatureImage(year,featureImage,MINVAL,MAXVAL);
        classifyYears.push_back(std::make_pair(year,featureImage));
    }
    vector<vector<LandCover>> trueClasses2022;
    ReadTrueClasses(trueClasses2022);
    vector<urban_Sample> testDataset;
    StudyTrueClasses(trueClasses2022,classifyYears.back().second,testDataset,100);

    std::cout<<"start classify"<<std::endl;
    // classifiy series
    vector<std::shared_ptr<Classified>> imageSeries;
    for (vector<YearImage>::iterator it = classifyYears.begin(); it != classifyYears.end(); it++){
        imageSeries.push_back(std::move(ClassifySingleYear(trainDataset,trainDataset,*it,classifierForUse)));
        std::cout<<it->first<<" finished."<<std::endl;
    }
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
std::shared_ptr<Classified> ClassifySingleYear( const vector<urban_Sample>& trainDataset,
                                                const vector<urban_Sample>& testDataset,
                                                const YearImage& yearImage,
                                                const vector<std::string>& classifierForUse){
    using BaseClassifier = T_Classifier<LandCover>;
    std::shared_ptr<Classified> classified = std::make_shared<Classified>();
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
        classifier->Train(trainDataset);
        classMat singleYearPixelClasses;
        //classifier->Classify(yearImage.second,singleYearPixelClasses,LandCover::Edge,LandCover::UNCLASSIFIED,MINVAL,MAXVAL,classifierKernelSize,false);
        //pixelClasses.push_back(singleYearPixelClasses);
        //classifiers.push_back(std::unique_ptr<BaseClassifier>(classifier));
        cv::Mat testImage;
        classifier->Classify(yearImage.second,testImage,LandCover::Edge,LandCover::UNCLASSIFIED,MINVAL,MAXVAL,classifierKernelSize,classifyColor,false);
        classifier->Examine(testDataset);
        classifier->Print(testImage,classFolderNames,"./"+std::to_string(yearImage.first));
        classified->setImage(testImage);
        std::cout<<yearImage.first<<'-'<<classifier->getName()<<" finished."<<std::endl;
    }
    CombinedClassifier(classified,classifiers,pixelClasses,yearImage.second,MINVAL,MAXVAL,classifierKernelSize,classifyColor);
    classified->CalcUrbanMorphology({classifyColor[LandCover::Imprevious],classifyColor[LandCover::Bareland]});
    classified->Examine(testDataset);
    return classified;
}
bool SeriesAnalysis(const vector<std::shared_ptr<Classified>>& imageSeries,
                    vector<double>& increasingRate,
                    vector<char>& increasingDirection){
    increasingRate.clear();
    increasingDirection.clear();
    vector<std::shared_ptr<Classified>>::const_iterator image = imageSeries.begin();
    double lastArea = (*image)->getArea();
    cv::Mat lastImage = (*image)->getUrbanMask().clone();
    ++image;
    for (;image != imageSeries.end(); image++){
        double currentArea = (*image)->getArea();
        cv::Mat currentImage = (*image)->getUrbanMask().clone();
        increasingRate.push_back((currentArea - lastArea) / lastArea);
        increasingDirection.push_back(UrbanMaskAnalysis(lastImage,currentImage));
        lastArea = currentArea;
        lastImage = currentImage.clone();
    }
    return true;
}
bool ReadTrueClasses(classMat& trueClasses2022){
    using namespace ningbo;
    cv::Mat classifiedImage = cv::imread("../landuse/ningbo/true_classified-clip.tif",cv::IMREAD_UNCHANGED);
    classifiedImage.convertTo(classifiedImage,CV_8U);
    for (int j = 0; j < classifiedImage.rows; j++){
        vClasses trueClassesRow;
        for (int i = 0; i < classifiedImage.cols; i++){
            uchar value = classifiedImage.at<uchar>(j,i);
            if (value == 10 || value == 30)
                trueClassesRow.push_back(LandCover::Greenland);
            else if (value == 40)
                trueClassesRow.push_back(LandCover::CropLand);
            else if (value == 50)
                trueClassesRow.push_back(LandCover::Imprevious);
            else if (value == 60)
                trueClassesRow.push_back(LandCover::Bareland);
            else if (value == 80)
                trueClassesRow.push_back(LandCover::Water);
            else
                trueClassesRow.push_back(LandCover::UNCLASSIFIED);
        }
        trueClasses2022.push_back(trueClassesRow);
    }
    cv::Mat classified = cv::Mat::zeros(classifiedImage.rows, classifiedImage.cols, CV_8UC3);
    classified.setTo(cv::Vec3b(255,255,255));
    for (int y = 0; y < trueClasses2022.size(); y++)
        for (int x = 0; x < trueClasses2022[y].size(); x++)
            classified.at<cv::Vec3b>(y, x) = classifyColor[trueClasses2022[y][x]];
    cv::imwrite("./true_classified.png", classified);
    //cv::imshow("True classes", classified);
    //cv::waitKey(0);
    //cv::destroyWindow("True classes");
    return true;
}
bool CombinedClassifier(std::shared_ptr<Classified> classified,
                        const vector<std::unique_ptr<T_Classifier<LandCover>>>& classifiers,
                        const vector<vector<vector<LandCover>>> pixelClasses,
                        const cv::Mat& featureImage,
                        const vFloat& minVal,const vFloat& maxVal,int classifierKernelSize,
                        const std::unordered_map<LandCover,cv::Vec3b>& classifyColor){
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
    classifiedImage.setTo(classifyColor.at(LandCover::UNCLASSIFIED));
    int y = 0;
    for (typename ClassMat::const_iterator row = combinedClasses.begin(); row != combinedClasses.end(); row++,y+=classifierKernelSize/2){
        int x = 0;
        for (typename vClasses::const_iterator col = row->begin(); col != row->end(); col++,x+=classifierKernelSize/2){
            if (x >= featureImage.cols - classifierKernelSize/2)
                break;
            cv::Rect window(x,y,classifierKernelSize/2,classifierKernelSize/2);
            classifiedImage(window) = classifyColor.at(*col);
            cv::Vec3b color = classifyColor.at(*col);
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
    std::string suitFolderPath = "../landuse/ningbo/sampling/";
    for (unsigned int classID = 0; classID < LandCover::CoverType; classID++){
        std::string classFolderPath = suitFolderPath + classFolderNames[static_cast<LandCover>(classID)] + "/";
        //std::cout<<classFolderPath<<std::endl;
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
        MINVAL[i] = 65535;
    }
    for (unsigned int classID = 0; classID < LandCover::CoverType; classID++){
        const std::vector<vFloat>& avg = classParas[classID].getAvg();
        const std::vector<vFloat>& var = classParas[classID].getVar();
        const unsigned int recordNum = classParas[classID].getRecordsNum();
        for (unsigned int i = 0; i < recordNum; i++){
            vFloat data;
            for (unsigned int d = 0; d < Spectra::SpectralNum; d++){
                //MAXVAL[d * 2] = std::max(MAXVAL[d],avg[i][d]);
                //MINVAL[d * 2] = std::min(MINVAL[d],avg[i][d]);
                //MAXVAL[d * 2 + 1] = std::max(MAXVAL[d],var[i][d]);
                //MINVAL[d * 2 + 1] = std::min(MINVAL[d],var[i][d]);
                data.push_back(avg[i][d]);
                //data.push_back(var[i][d]);
            }
            //bool isTrain = (rand()%10) <= (trainRatio*10);
            bool isTrain = i <= (recordNum * trainRatio);
            urban_Sample sample(static_cast<LandCover>(classID),data,isTrain);
            dataset.push_back(sample);
        }
    }
    //for (std::vector<urban_Sample>::iterator data = dataset.begin(); data != dataset.end(); data++)
    //    data->scalingFeatures(MAXVAL,MINVAL);
    return true;
}
bool StudyTrueClasses(const vector<vector<LandCover>>&trueClasses2022,const cv::Mat& RSimage,vector<urban_Sample>& trainDataset,size_t setSize){
    int row = RSimage.rows, col = RSimage.cols;
    srand(time(0));
    int counter = setSize;
    int sampleRange = classifierKernelSize / 2;
    while (counter){
        int y = rand()%(row - sampleRange * 2) + sampleRange;
        int x = rand()%(col - sampleRange * 2) + sampleRange;
        LandCover coverType = trueClasses2022[y][x];
        for (int i = 0; i < sampleRange; i++)
            for (int j = 0; j < sampleRange; j++)
                if (trueClasses2022[y + i][x + j] != coverType)
                    continue;
        if (coverType == LandCover::UNCLASSIFIED)
            continue;
        cv::Rect window(x,y,sampleRange,sampleRange);
        vFloat data;
        cv::Mat sample = RSimage(window);
        std::vector<cv::Mat> channels;
        cv::split(sample,channels);
        bool blank = false;
        for (int d = 0; d < Spectra::SpectralNum; d++){
            int count = 0;
            float val = 0.0f;
            for (int i = 0; i < sampleRange; i++)
                for (int j = 0; j < sampleRange; j++){
                    float value = channels[d].at<ushort>(y+i,x+j);
                    if (value > 0 && value < 65535){
                        val += value;
                        ++count;
                    }
                }
            if (count == 0)
                blank = true;
            else
                data.push_back(val / count);
        }
        if (blank)
            continue;
        --counter;
         urban_Sample urbanSample(coverType,data,true);
        trainDataset.push_back(urbanSample);
    }
    return true;
}
}//namespace ningbo