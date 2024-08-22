#ifndef MYCONTOUR_H
#define MYCONTOUR_H

#include <opencv2/core/types.hpp>

struct Contour {
    int parent{};
    int id{};
    bool isOuterBorder{};
    std::vector<cv::Point> points;
};

void my_findContour(const cv::Mat& srcImage, cv::Mat& dstImage, std::vector<Contour>& contours);

void convert_to_binary(const cv::Mat& srcImage, cv::Mat& dstImage);

// normally in contour points x is row, y is column index, but needed to swap x and y to use other functions
void swap_x_and_y(const Contour& contour, Contour& reversed_contour);

void contour_threshold(const std::vector<std::vector<cv::Point>>& srcContours, std::vector<std::vector<cv::Point>>& dstContours, int threshold);



#endif // MYCONTOUR_H
