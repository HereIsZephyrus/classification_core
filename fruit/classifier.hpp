#ifndef CLASSFIERHPP
#define CLASSFIERHPP
#include "../func.hpp"
#include "../t_classifier.hpp"
struct Border{
    int label;
    int count;
    long long centerX,centerY;
    bool operator <(const Border &b) const {
        return count > b.count;
    }
    Border(int label,int count){
        this->label = label;
        this->count = count;
        this->centerX = -1;
        this->centerY = -1;
    }
};
enum Classes : unsigned int{
    Desk,
    Apple,
    Blackplum,
    Dongzao,
    Grape,
    Peach,
    Yellowpeach,
    counter,
    Edge,
    Unknown
};
extern std::string classFolderNames[Classes::counter];
extern std::unordered_map<Classes,cv::Scalar> classifyColor;
enum Demisions : unsigned int{
    hue,
    saturation,
    value,
    //gradient,
    angle,
    dim
};
typedef std::vector<Classes> vClasses;
typedef T_StaticPara<Classes> StaticPara;
typedef T_Sample<Classes> Sample;
namespace bayes{
typedef T_NaiveBayesClassifier<Classes> NaiveBayesClassifier;
typedef T_NonNaiveBayesClassifier<Classes> NonNaiveBayesClassifier;   
}
namespace linear{
typedef T_FisherClassifier<Classes> FisherClassifier;
bool LinearClassify(const cv::Mat& rawimage,FisherClassifier* classifer,std::vector<std::vector<Classes>>& patchClasses); 
}
#endif