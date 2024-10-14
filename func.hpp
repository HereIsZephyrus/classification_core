#ifndef FUNC_HPP
#define FUNC_HPP
#include <opencv2/opencv.hpp>
#include <cstring>
#include <string>
#define SHOW_WINDOW false
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
enum Classes{
    Desk,
    Apple,
    Blackplum,
    Dongzao,
    Grape,
    Peach,
    Yellowpeach,
    counter
};
extern std::string classFolderNames[Classes::counter];
enum Demisions{
    blue,
    green,
    red,
    gray,
    gradient,
    dim
};
constexpr int classifierKernelSize = 5;

namespace tcb{
bool TopHat(cv::Mat &image,int xSize,int ySize);
bool BottomHat(cv::Mat &image,int xSize,int ySize);
bool GaussianSmooth(cv::Mat &image,int xSize,int ySize,int sigma);
bool rgb2gray(cv::Mat &image);
bool Sobel(cv::Mat &image,int dx,int dy,int bandwidth);
bool Laplacian(cv::Mat &image,int bandwidth);
bool BoxSmooth(cv::Mat &image,int xSize,int ySize);
bool Erode(cv::Mat &image,int kernelSize);
bool Dilate(cv::Mat &image,int kernelSize);
bool drawCircleDDA(cv::Mat &image, int h, int k, float rx,float ry);
};
namespace bayers{
class StaticPara{
    std::vector<std::vector<float>> mu,sigma;
    Classes classID;
    unsigned int recordNum;
public:
    StaticPara() = default;
    StaticPara(Classes classID):recordNum(0),classID(classID){
        mu.reserve(Demisions::dim);
        sigma.reserve(Demisions::dim);
    }
    ~StaticPara(){}
    void InitClassType(Classes ID);
    void Sampling(const std::string& entryPath);
};

}
#endif