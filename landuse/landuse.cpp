#include "landuse.hpp"
#include "image.hpp"
int LanduseMain(){
    cv::Mat rawImage;
    generateFeatureImage(rawImage);
    {
        StaticPara* classParas = new StaticPara[classesNum];
        for (unsigned int classID = 0; classID < classesNum; classID++)
            classParas[classID].InitClassType(static_cast<Classes>(classID));
        std::vector<Sample> dataset;
        StudySamples(classParas,dataset);
        delete[] classParas;
    }
    land_NaiveBayesClassifier* bayes = new land_NaiveBayesClassifier();
    bayes->Train(dataset);
    bayes->Classify(rawImage);
    bayes->print();
    land_FisherClassifier* fisher = new land_FisherClassifier();
    fisher->Train(dataset);
    fisher->Classify(rawImage);
    fisher->print();
    land_SVMClassifier* svm = new land_SVMClassifier();
    svm->Train(dataset);
    svm->Classify(rawImage);
    svm->print();
    land_FCNClassifier* fcn = new land_FCNClassifier();
    fcn->Train(dataset);
    fcn->Classify(rawImage);
    fcn->print();
    land_RandomClassifier* randomforest = new land_RandomClassifier();
    randomforest->Train(dataset);
    randomforest->Classify(rawImage);
    randomforest->print();
    return 0;
}
int SeriesMain(){
    return 0;
}