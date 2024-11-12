#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <random>
#include <algorithm>
#include "rs_classifier.hpp"
using namespace Eigen;
template<>
void land_StaticPara::InitClassType(weilaicheng::LandCover ID){
    recordNum = 0;
    classID = ID;
    avg.clear();
    var.clear();
    return;
}
template<>
void land_StaticPara::Sampling(const std::string& entryPath){
    const int classifierKernelSize = defaultClassifierKernelSize;
    using namespace weilaicheng;
    std::vector<cv::Mat> channels;
    cv::imreadmulti(entryPath,channels,cv::IMREAD_UNCHANGED);
    weilaicheng::GenerateFeatureChannels(channels);
    const unsigned int patchRows = channels[0].rows, patchCols = channels[0].cols;
    for (unsigned int left = 0; left < patchCols - classifierKernelSize + 1; left+=classifierKernelSize/2){
        for (unsigned int top = 0; top < patchRows - classifierKernelSize + 1; top+=classifierKernelSize/2){
            cv::Rect window(left, top, classifierKernelSize, classifierKernelSize);
            vFloat means, vars;
            for (unsigned int i = 0; i < weilaicheng::Spectra::SpectralNum; i++){
                cv::Mat viewingPatch = channels[i](window);
                cv::Scalar mean, stddev;
                cv::meanStdDev(viewingPatch, mean, stddev);
                means.push_back(mean[0]);
                vars.push_back(stddev[0] * stddev[0]);
            }
            avg.push_back(means);
            var.push_back(vars);
            recordNum++;
        }
    }
    return;
}
namespace weilaicheng{
std::string classFolderNames[LandCover::CoverType] = 
{"Water","Greenland","bareland","Imprevious"};
std::unordered_map<LandCover,cv::Scalar> classifyColor = {
    {LandCover::Water,cv::Scalar(255,0,0)}, // blue
    {LandCover::Imprevious,cv::Scalar(0,0,255)}, // red
    {LandCover::Bareland,cv::Scalar(42,42,165)}, // brzone,
    {LandCover::Greenland,cv::Scalar(0,255,0)}, // green
    {LandCover::Edge,cv::Scalar(255,255,255)}, // white
    {LandCover::UNCLASSIFIED,cv::Scalar(255,255,255)}, // black
};
void land_NaiveBayesClassifier::Train(const std::vector<land_Sample>& dataset,const float* classProbs){
    featureNum = dataset[0].getFeatures().size(); //select all
    para.clear();
    unsigned int classNum = getClassNum();
    std::vector<double> classifiedFeaturesAvg[classNum],classifiedFeaturesVar[classNum];
    for (int i = 0; i < classNum; i++){
        classifiedFeaturesAvg[i].assign(featureNum,0.0);
        classifiedFeaturesVar[i].assign(featureNum,0.0);
    }
    std::vector<size_t> classRecordNum(classNum,0);
    for (std::vector<land_Sample>::const_iterator it = dataset.begin(); it != dataset.end(); it++){
        unsigned int label = static_cast<unsigned int>(it->getLabel());
        const vFloat& sampleFeature = it->getFeatures();
        for (unsigned int i = 0; i < featureNum; i++)
            classifiedFeaturesAvg[label][i] += sampleFeature[i];
        classRecordNum[label]++;
    }
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < featureNum; j++)
            classifiedFeaturesAvg[i][j] /= classRecordNum[i];
    for (std::vector<land_Sample>::const_iterator it = dataset.begin(); it != dataset.end(); it++){
        unsigned int label = static_cast<unsigned int>(it->getLabel());
        const vFloat& sampleFeature = it->getFeatures();
        for (unsigned int i = 0; i < featureNum; i++)
            classifiedFeaturesVar[label][i] += (sampleFeature[i] - classifiedFeaturesAvg[label][i]) * (sampleFeature[i] - classifiedFeaturesAvg[label][i]);
    }
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < featureNum; j++)
            classifiedFeaturesVar[i][j] = std::sqrt(classifiedFeaturesVar[i][j]/classRecordNum[i]);
    for (unsigned int i = 0; i < classNum; i++){
        BasicParaList temp;
        temp.w = classProbs[i];
        temp.mu = classifiedFeaturesAvg[i];
        temp.sigma = classifiedFeaturesVar[i];
        para.push_back(temp);
        {
            std::cout<<"class "<<i<<" counts"<<classRecordNum[i]<<std::endl;
            std::cout<<"w: "<<temp.w<<std::endl;
            std::cout<<"mu: ";
            for (unsigned int j = 0; j < featureNum; j++)
                std::cout<<temp.mu[j]<<" ";
            std::cout<<std::endl;
            std::cout<<"sigma: ";
            for (unsigned int j = 0; j < featureNum; j++)
                std::cout<<temp.sigma[j]<<" ";
            std::cout<<std::endl;
        }
    }
    return;
}
bool land_NaiveBayesClassifier::CalcClassProb(float* prob){
    using namespace weilaicheng;
    unsigned int* countings = new unsigned int[LandCover::CoverType];
    unsigned int totalRecord = 0;
    for (int i = 0; i < LandCover::CoverType; i++)
        countings[i] = 0;
    std::string filename = "../landuse/sampling/classification.csv";
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "can't open file!" << filename << std::endl;
        return false;
    }
    std::getline(file, line);// throw header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;
        while (std::getline(ss, value, ',')) {}
        totalRecord++;
        if (value == "Water")
            countings[LandCover::Water]++;
        else if (value == "Greenland")
            countings[LandCover::Greenland]++;
        else if (value == "Bareland")
            countings[LandCover::Bareland]++;
        else if (value == "Imprevious")
            countings[LandCover::Imprevious]++;
    }
    file.close();
    for (int i = 0; i < LandCover::CoverType; i++)
        prob[i] = static_cast<float>(countings[i]) / totalRecord;
    delete[] countings;
    return true;
}
void land_NaiveBayesClassifier::Train(const std::vector<land_Sample>& dataset){
    unsigned int classesNum = getClassNum();
    float* classProbs = new float[classesNum];
    CalcClassProb(classProbs);
    Train(dataset,classProbs);
    delete[] classProbs;
}
void land_FisherClassifier::Train(const std::vector<land_Sample>& dataset){
    featureNum = dataset[0].getFeatures().size(); //select all
    fMat SwMat,SbMat;
    SwMat = new float*[featureNum];
    SbMat = new float*[featureNum];
    for (size_t i = 0; i < featureNum; i++){
        SwMat[i] = new float[featureNum];
        SbMat[i] = new float[featureNum];
        for (size_t j = 0; j < featureNum; j++)
            SwMat[i][j] = 0.0f,SbMat[i][j] = 0.0f;
    }
    CalcSwSb(SwMat,SbMat,dataset);
    MatrixXf Sw(featureNum,featureNum),Sb(featureNum,featureNum);
    for (size_t i = 0; i < featureNum; i++)
        for (size_t j = 0; j < featureNum; j++)
            Sw(i,j) = SwMat[i][j],Sb(i,j) = SbMat[i][j];
    for (size_t i = 0; i < featureNum; i++){
        delete[] SwMat[i];
        delete[] SbMat[i];
    }
    delete[] SwMat;
    delete[] SbMat;
    SelfAdjointEigenSolver<MatrixXf> eig(Sw.completeOrthogonalDecomposition().pseudoInverse() * Sb);
    int classNum = getClassNum();
    MatrixXf projectionMatrix = eig.eigenvectors().rightCols(1);
    for (size_t j = 0; j < featureNum; j++){
        projMat.push_back(projectionMatrix(j));
    }
    for (int i = 0; i < classNum; i++){
        float calcmean = 0;
        for (size_t j = 0; j < featureNum; j++)
            calcmean += projectionMatrix(j) * mu[i][j];
        signal.push_back(calcmean);
        std::cout<<signal[i]<<std::endl;
    }
    return;
}
void land_SVMClassifier::Train(const std::vector<land_Sample>& dataset){
    featureNum = dataset[0].getFeatures().size(); //select all
    unsigned int classNum = getClassNum();
    std::vector<vFloat> waterPN,classCount[classNum];
    std::vector<int> waterLabel;
    int selectTag = 0;
    for (std::vector<land_Sample>::const_iterator it = dataset.begin(); it != dataset.end(); it++){
        unsigned int classID = static_cast<unsigned int>(it->getLabel());
        if (it->getLabel() == LandCover::Water){
            waterPN.push_back(it->getFeatures());
            waterLabel.push_back(0);
        }else{
            if (!selectTag){
                waterPN.push_back(it->getFeatures());
                waterLabel.push_back(1);
            }
            selectTag = (selectTag+1)%3;
        }
        classCount[classID].push_back(it->getFeatures());
    }
    OVOSVM waterClassifier = OVOSVM(LandCover::Water,LandCover::UNCLASSIFIED);
    waterClassifier.fit(waterPN,waterLabel);
    classifiers.push_back(waterClassifier);
    unsigned int waterID = static_cast<unsigned int>(LandCover::Water);
    for (unsigned int i = 0; i < classNum; i++)
        for (unsigned int j = 0; j < classNum; j++){
            if (i == waterID || j == waterID)
                continue;
            std::vector<vFloat> classPN = classCount[i];
            std::vector<int> classLabeli,classLabelj;
            classLabeli.assign(classCount[i].size(),0);
            classLabelj.assign(classCount[j].size(),1);
            classLabeli.insert(classLabeli.end(), classLabelj.begin(), classLabelj.end());
            classPN.insert(classPN.end(), classCount[j].begin(), classCount[j].end());
            OVOSVM classifier = OVOSVM(static_cast<LandCover>(i),static_cast<LandCover>(j));
            classifier.fit(classPN,classLabeli);
            classifiers.push_back(classifier);
        }
}
void land_BPClassifier::Train(const std::vector<land_Sample>& dataset){
    featureNum = dataset[0].getFeatures().size(); //select all
    classNum = getClassNum();
    weights.resize(featureNum);
    srand(time(0));
    for (int i = 0; i < featureNum; ++i)
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    for (int epoch = 0; epoch < 10000; ++epoch)
        for (std::vector<land_Sample>::const_iterator it = dataset.begin(); it != dataset.end(); ++it)
            backward(it->getFeatures(), it->getLabel());
}
void land_RandomClassifier::Train(const std::vector<land_Sample>& dataset){
    for (int i = 0; i < nTrees; ++i) {
        DecisionTree tree(LandCover::UNCLASSIFIED,maxDepth);
        std::vector<vFloat> bootstrapX;
        std::vector<LandCover> bootstrapY;
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0, dataset.size() - 1);
        for (std::vector<land_Sample>::const_iterator data = dataset.begin(); data != dataset.end(); data++) {
            int index = distribution(generator);
            bootstrapX.push_back(data->getFeatures());
            bootstrapY.push_back(data->getLabel());
        }
        tree.fit(bootstrapX, bootstrapY);
        trees.push_back(tree);
    }
}
bool GenerateClassifiedImage(const cv::Mat& rawimage,cv::Mat& classified,const std::vector<std::vector<LandCover>>& pixelClasses){
    using ClassMat = std::vector<std::vector<LandCover>>;
    using vClasses = std::vector<LandCover>;
    const int classifierKernelSize = defaultClassifierKernelSize;
    classified = cv::Mat::zeros(rawimage.rows, rawimage.cols, CV_8UC3);
    classified.setTo(classifyColor[LandCover::UNCLASSIFIED]);
    int y = 0;
    for (ClassMat::const_iterator row = pixelClasses.begin(); row != pixelClasses.end(); row++,y+=classifierKernelSize/2){
        int x = 0;
        for (vClasses::const_iterator col = row->begin(); col != row->end(); col++,x+=classifierKernelSize/2){
            if (x >= rawimage.cols - classifierKernelSize/2)
                break;
            cv::Rect window(x,y,classifierKernelSize/2,classifierKernelSize/2);
            classified(window) = classifyColor[*col];
        }
        if (y >= rawimage.rows - classifierKernelSize/2)
            break;
    }
    return true;
}
}