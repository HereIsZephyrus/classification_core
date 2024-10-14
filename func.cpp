#include "func.hpp"

std::string classFolderNames[Classes::counter] = {"desk","apple","blackplum","dongzao","grape","peach","yellowpeach"};
namespace tcb{
bool TopHat(cv::Mat &image,int xSize,int ySize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(xSize, ySize));
    cv::morphologyEx(image, image, cv::MORPH_OPEN, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Top Hat Transform", image);
        cv::waitKey(0);
        cv::destroyWindow("Top Hat Transform");
    }
    return true;
}
bool BottomHat(cv::Mat &image,int xSize,int ySize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(xSize, ySize));
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Bottom Hat Transform", image);
        cv::waitKey(0);
        cv::destroyWindow("Bottom Hat Transform");
    }
    return true;
}
bool GaussianSmooth(cv::Mat &image,int xSize,int ySize,int sigma){
    cv::GaussianBlur(image, image, cv::Size(xSize, ySize), 0);
    if (SHOW_WINDOW){
        cv::imshow("Smoothed Image with Gaussian Blur", image);
        cv::waitKey(0);
        cv::destroyWindow("Smoothed Image with Gaussian Blur");
    }
    return true;
}
bool rgb2gray(cv::Mat &image){
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    if (SHOW_WINDOW){
        cv::imshow("Gray Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Gray Image");
    }
    return true;
}
bool Sobel(cv::Mat &image,int dx,int dy,int bandwidth){
    cv::Mat sobeled;
    cv::Sobel(image, sobeled, CV_64F, dx, dy, bandwidth);
    cv::convertScaleAbs(sobeled, image);
    if (SHOW_WINDOW){
        cv::imshow("Sobel Image (Vertical Lines)", image);
        cv::waitKey(0);
        cv::destroyWindow("Sobel Image (Vertical Lines)");
    }
    return true;
}

bool Laplacian(cv::Mat &image,int bandwidth){
    cv::Mat laplacianImage;
    cv::Laplacian(image, laplacianImage, CV_16S, bandwidth);
    cv::Mat absLaplacianImage;
    cv::convertScaleAbs(laplacianImage, image);
    if (SHOW_WINDOW){
        cv::imshow("Laplacian Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Laplacian Image");
    }
    return true;
}

bool BoxSmooth(cv::Mat &image,int xSize,int ySize){
    cv::blur(image, image, cv::Size(xSize, ySize));
    if (SHOW_WINDOW){
        cv::imshow("Smoothed Image with Box Blur", image);
        cv::waitKey(0);
        cv::destroyWindow("Smoothed Image with Box Blur");
    }
    return true;
}
bool Erode(cv::Mat &image,int kernelSize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::erode(image, image, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Eroded Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Eroded Image");
    }
    return true;
}
bool Dilate(cv::Mat &image,int kernelSize){
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::dilate(image, image, kernel);
    if (SHOW_WINDOW){
        cv::imshow("Dilated Image", image);
        cv::waitKey(0);
        cv::destroyWindow("Dilated Image");
    }
    return true;
}
bool drawCircleDDA(cv::Mat &image, int h, int k, float rx,float ry) {
    int lineThickness = 8;
    for (int theta = 0; theta < 3600; theta++) {
        int x = static_cast<int>(h + rx * cos(theta/10 * CV_PI / 180));
        int y = static_cast<int>(k + ry * sin(theta/10 * CV_PI / 180));
        for (int i = -lineThickness; i <= lineThickness; i++)
            for (int j = -lineThickness; j <= lineThickness; j++)
                if (x + i >= 0 && x + i < image.cols && y + j >= 0 && y + j < image.rows)
                    image.at<uchar>(y+j,x+i) = 255;
    }
    return true;
}
};
namespace bayers {
void StaticPara::InitClassType(Classes ID){
    recordNum = 0;
    classID = ID;
    mu.clear();
    sigma.clear();
    mu.reserve(Demisions::dim);
    sigma.reserve(Demisions::dim);
    return;
}
void StaticPara::Sampling(const std::string& entryPath){
    cv::Mat patch = cv::imread(entryPath);
    if (patch.empty()){
        std::cerr << "Image not found!" << std::endl;
        return;
    }
    std::vector<cv::Mat> channels;
    cv::split(patch, channels);
    cv::Mat grayImage;
    cv::cvtColor(patch, grayImage, cv::COLOR_BGR2GRAY);
    channels.push_back(grayImage);
    cv::Mat sobelx,sobely,magnitude,angle;
    cv::Sobel(patch, sobelx, CV_64F, 1, 0, 3);
    cv::Sobel(patch, sobely, CV_64F, 0, 1, 3);
    cv::cartToPolar(sobelx, sobely, magnitude, angle, true);
    channels.push_back(magnitude);
    const unsigned int patchRows = patch.rows, patchCols = patch.cols;
    for (unsigned int left = 0; left < patchCols - classifierKernelSize; left+=classifierKernelSize){
        for (unsigned int top = 0; top < patchRows - classifierKernelSize; top+=classifierKernelSize){
            cv::Rect window(left, top, classifierKernelSize, classifierKernelSize);
            for (unsigned int i = 0; i < Demisions::dim; i++){
                cv::Mat viewingChannel = channels[i];
                cv::Mat viewingPatch = viewingChannel(window);
                cv::Scalar mean, stddev;
                cv::meanStdDev(viewingPatch, mean, stddev);
                mu[i].push_back(mean[0]);
                sigma[i].push_back(stddev[0]);
                std::cout<<i<<' '<<mean[0]<<" "<<stddev[0]<<std::endl;
            }
        }
    }
    return;
}
};