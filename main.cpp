#include <iostream>
#include <fstream>
#include "globalmatting.cpp"
#include "guidedfilter.cpp"
#include <omp.h>

using namespace std;

int main(int argc, char **argv)
{
    //cv::Mat image = cv::imread("../data/GT04-image.png", CV_LOAD_IMAGE_COLOR);
    //cv::Mat trimap = cv::imread("../data/GT04-trimap.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat trimap = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    // (optional) exploit the affinity of neighboring pixels to reduce the 
    // size of the unknown region. please refer to the paper
    // 'Shared Sampling for Real-Time Alpha Matting'.
    int64 start = cv::getTickCount();
    expansionOfKnownRegions(image, trimap, 9);
    double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    cout << "Expansion : " << timeSec << " sec" << endl;

    cv::Mat foreground, alpha;
    start = cv::getTickCount();
    globalMatting(image, trimap, foreground, alpha);
    timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    cout << "Matting : " << timeSec << " sec" << endl;
    cv::imwrite(argv[3], alpha);
    cv::imwrite(argv[4], foreground);
    // filter the result with fast guided filter
    start = cv::getTickCount();
    alpha = guidedFilter(image, alpha, 10, 1e-5);
#pragma omp parallel for
    for (int x = 0; x < trimap.cols; ++x)
        for (int y = 0; y < trimap.rows; ++y)
        {
            if (trimap.at<uchar>(y, x) == 0)
                alpha.at<uchar>(y, x) = 0;
            else if (trimap.at<uchar>(y, x) == 255)
                alpha.at<uchar>(y, x) = 255;
        }
    timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    cout << "GuidedFiltering : " << timeSec << " sec" << endl;
    cv::imwrite(argv[5], alpha);
    return 0;
    
}
