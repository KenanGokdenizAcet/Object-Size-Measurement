#include <opencv2/core/types.hpp>
#include <vector>

#ifndef MYCONVEXHULL_H
#define MYCONVEXHULL_H


void my_convex_hull(const std::vector<cv::Point>& srcPoints, std::vector<cv::Point>& convex_points);
void my_sort(const std::vector<cv::Point>& points, std::vector<cv::Point>& sorted_points, int mode);

#endif // MYCONVEXHULL_H
