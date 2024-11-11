#include <cmath>
#include <algorithm>
#include "func.hpp"
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
bool CalcChannelMeanStds(const std::vector<cv::Mat> & channels, vFloat & data){
    data.clear();
    for (std::vector<cv::Mat>::const_iterator it = channels.begin(); it != channels.end(); it++){
        cv::Scalar mean, stddev;
        cv::meanStdDev(*it, mean, stddev);
        data.push_back(cv::mean(*it)[0]);
        data.push_back(stddev[0] * stddev[0]);
    }
    return true;
}
bool CalcInvMat(float** const convMat,float ** invMat,int num){
    fMat augmented = new float*[num];
    for (size_t i = 0; i < num; i++)
        augmented[i] = new float[num*2];
    for (size_t i = 0; i < num; i++) {
        for (size_t j = 0; j < num; j++) 
            augmented[i][j] = convMat[i][j];
        for (size_t j = num; j < 2 * num; j++)
            if (i == (j - num))
                augmented[i][j] = 1.0;
            else
                augmented[i][j] = 0.0;
    }
    for (size_t i = 0; i < num; i++) {
        double pivot = augmented[i][i];
        for (size_t j = 0; j < 2 * num; j++)
            augmented[i][j] /= pivot;
        for (size_t k = 0; k < num; ++k) {
            if (k == i) 
                continue;
            double factor = augmented[k][i];
            for (size_t j = 0; j < 2 * num; j++) 
                augmented[k][j] -=  factor * augmented[i][j];
        }
    }
    for (size_t i = 0; i < num; i++)
        for (size_t j = 0; j < num; j++)
            invMat[i][j] = augmented[i][j + num];
    if (augmented != nullptr){
        for(size_t i = 0;i < num;i++)
            delete[] augmented[i];
        delete[] augmented;
    }
    return true;
}
bool CalcEigen(const std::vector<vFloat>& matrix, vFloat& eigVal, std::vector<vFloat>& eigVec, const int num){
    std::vector<vFloat> Q(num,vFloat(num,0.0f));
    std::vector<vFloat> R(num,vFloat(num,0.0f));
    std::vector<vFloat> A(num,vFloat(num,0.0f));
    for (int i = 0; i < num; i++)
        eigVec[i][i] = 1.0f;
    A = matrix;
    const int maxIter = 3;
    const float tolerance = 1e-6;
    for (int iter = 0; iter < maxIter; iter++){
        //QR decomposition
        for (int j = 0; j < num; j++) {
            for (int i = 0; i < num; i++)
                Q[i][j] = A[i][j];
            for (int i = 0; i < j; ++i) {
                float temp = 0;
                for (size_t k = 0; k < num; ++k)
                    temp += Q[k][j] * A[k][i];
                R[i][j] = temp;
                for (int k = 0; k < num; ++k)
                    Q[k][j] -= R[i][j] * Q[k][i];
            }
            {
                float temp = 0;
                for (size_t i = 0; i < num; ++i)
                    temp += Q[i][j] * Q[i][j];
                R[j][j] = sqrt(temp);
            }
            for (int i = 0; i < num; ++i)
                Q[i][j] /= R[j][j];
        }

        //update A by QR
        //std::cout<<"iter "<<iter<<std::endl;
        for (int i = 0; i < num; i++){
            for (int j = 0; j < num; j++) {
                A[i][j] = 0.0f;
                for (int k = 0; k < num; k++)
                    A[i][j] += R[k][j] * Q[i][k];
                //std::cout<<A[i][j]<<' ';
            }
            //std::cout<<std::endl;
        }
        
        //update eigVec by Q
        std::vector<vFloat> temp(num,vFloat(num,0.0f));
        for (int i = 0; i < num; i++)
            for (int j = 0; j < num; j++)
                for (int k = 0; k < num; k++){
                    temp[i][j] += eigVec[i][k] * Q[k][j];
                    //std::cout<<temp[i][j]<<"+="<<eigVec[i][k]<<'*'<<Q[k][j]<<' ';
                }
        eigVec = temp;
        for (int i = 1; i < num; i++)
            if (fabs(A[i][i - 1]) < tolerance)
                break;
    }
    for (int i = 0; i < num; i++)
        eigVal[i] = A[i][i];
    return true;
}
}//namespace tcb
namespace bayes{
double CalcConv(const vFloat& x, const vFloat& y){
    double res = 0;
    const size_t n = x.size();
    double xAvg = 0.0f, yAvg = 0.0f;
    for (size_t i = 0; i < n; i++){
        xAvg += x[i];
        yAvg += y[i];
    }
    xAvg /= n;    yAvg /= n;
    for (size_t i = 0; i < n; i++)
        res += (x[i] - xAvg) * (y[i] - yAvg);
    res /= (n-1);
    return res;
}
}//namespace bayes