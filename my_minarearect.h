#ifndef MY_MINAREARECT_H
#define MY_MINAREARECT_H

#include <opencv2/core/types.hpp>

void my_minAreaRect(const std::vector<cv::Point>& convex_hull, std::vector<cv::Point>& minAreaRect);


#endif // MY_MINAREARECT_H
