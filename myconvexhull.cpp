#include "myconvexhull.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

// compare col_row
struct compare_xy {
    bool operator() (const Point& left, const Point& right) const {
        return (left.x == right.x ? left.y < right.y : left.x < right.x);
    }
};

// compare_row_col
struct compare_yx {
    bool operator() (const Point& left, const Point& right) const {
        return (left.y == right.y ? left.x > right.x : left.y < right.y);
    }
};

void my_sort(const vector<Point>& points, vector<Point>& sorted_points, int mode) {
    sorted_points = points;
    if (mode == 0) {
        sort(sorted_points.begin(), sorted_points.end(), compare_xy());
    } else {
        sort(sorted_points.begin(), sorted_points.end(), compare_yx());
    }
}

/*
 * cross product of two vectors OA and OB
 *
 * returns positive for counter clockwise turn
 * returns negative for clockwise turn
 */
float cross_product(const Point O, const Point A, const Point B) {
    return (A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y);
}

void my_convex_hull(const vector<Point>& srcPoints, vector<Point>& convex_points) {
    vector<Point> sorted_points;
    my_sort(srcPoints, sorted_points, 0);
    size_t n { srcPoints.size() };
    Point p1 { sorted_points.at(0) }; // leftmost point
    Point p2 { sorted_points.at(n - 1) }; // rightmost point

    convex_points.push_back(p1);
    convex_points.push_back(p2);

    // build lower hull
    for (const auto& point : sorted_points) {
        while(convex_points.size() > 1 && cross_product(convex_points.at(convex_points.size()-2), convex_points.back(), point) <= 0) {
            convex_points.pop_back();
        }
        convex_points.push_back(point);
    }

    // build upper hull
    size_t lower_hull_size = convex_points.size();
    for (size_t i = n-2; i >= 1; i--) {
        while(convex_points.size() > lower_hull_size && cross_product(convex_points.at(convex_points.size()-2), convex_points.back(), sorted_points.at(i)) <= 0) {
            convex_points.pop_back();
        }
        convex_points.push_back(sorted_points.at(i));
    }

    convex_points.pop_back();
}
