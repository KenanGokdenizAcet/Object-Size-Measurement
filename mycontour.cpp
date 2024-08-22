#include "mycontour.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <ostream>

using namespace cv;
using namespace std;
using Pixel_value = int8_t;


namespace Colors{
    const cv::Scalar _Green{0, 255, 0},
                        Green{50, 255, 100},
                        Red{0, 0, 255},
                        _Blue{255, 0, 0},
                        Blue{155, 100,0},
                        Yellow{0,255,255},
                        Orange{0,155,255},
                        Purple{255,0,255};
}

constexpr Pixel_value Edge_value { 1 };
constexpr Pixel_value Non_edge_value { 0 };

const cv::Point Non_point(-1,-1);

enum class Dir {
    None = -1,
    Right,
    Up_right,
    Up,
    Up_left,
    Left,
    Down_left,
    Down,
    Down_right,
};

Pixel_value get_pixel_value(const Mat& img, const Point& p) {
    return img.at<Pixel_value>(p.x, p.y);
}

void set_pixel_value(Mat& img, Point p, Pixel_value value) {
    img.at<Pixel_value>(p.x, p.y) = value;
}

bool isOuterBorder(const Mat& img, const Point& p) {
     return get_pixel_value(img, p) == Edge_value
             && get_pixel_value(img, Point(p.x, p.y - 1)) == Non_edge_value;
}

bool isHoleBorder(const Mat& img, const Point p) {
    return get_pixel_value(img, p) >= Edge_value
            && get_pixel_value(img, Point(p.x,p.y + 1)) == Non_edge_value;
}

/*
 * point.x: row index
 * point.y: column index
 *
 */
Point next_point(const Point& p, const Dir& direction) {
    switch (direction) {
    case Dir::Right:
        return Point(p.x, p.y+1);
    case Dir::Down_right:
        return Point(p.x+1, p.y+1);
    case Dir::Down:
        return Point(p.x+1, p.y);
    case Dir::Down_left:
        return Point(p.x+1, p.y-1);
    case Dir::Left:
        return Point(p.x, p.y-1);
    case Dir::Up_left:
        return Point(p.x-1, p.y-1);
    case Dir::Up:
        return Point(p.x-1, p.y);
    case Dir::Up_right:
        return Point(p.x-1, p.y+1);
    default:
        cerr << "next_point -> Next point not handled!" << endl;
        abort();
    }
}

// get direction from the center to point p
Dir get_direction(const Point& center, const Point& p) {
    if (center.x == p.x && center.y > p.y) {
        return Dir::Left;
    } else if (center.x == p.x && center.y < p.y) {
        return Dir::Right;
    } else if (center.x < p.x && center.y == p.y) {
        return Dir::Down;
    } else if (center.x > p.x && center.y == p.y) {
        return Dir::Up;
    } else if (center.x > p.x && center.y < p.y) {
        return Dir::Up_right;
    } else if (center.x > p.x && center.y > p.y) {
        return Dir::Up_left;
    } else if (center.x < p.x && center.y < p.y) {
        return Dir::Down_right;
    } else if (center.x < p.x && center.y > p.y) {
        return Dir::Down_left;
    } else {
        cerr << "get_direction -> Direction not handled!\n";
        abort();
    }
}


Dir clockwiseNextDir(const Dir& curr_dir) {
    return static_cast<Dir>((static_cast<int>(curr_dir) + 7) % 8);
}

Dir cntrClockwiseNextDir(const Dir& curr_dir) {
    return static_cast<Dir>((static_cast<int>(curr_dir) + 1) % 8);
}

Point clockwiseSearch(const Mat& img, const Point& center, const Point& startPoint) {
    Dir curr_dir { get_direction(center, startPoint) };
    for (int i = 0; i < 8; i++) {

        Point point { next_point(center, curr_dir) };
        if (get_pixel_value(img, point) != Non_edge_value) {
            return point;
        }

        Dir next_dir { clockwiseNextDir(curr_dir) };
        curr_dir = next_dir;
    }
    return Non_point;
}

Point cntrClockwiseSearch(const Mat& img, const Point& center, const Point& startPoint, bool& isZeroExamined) {
    Dir curr_dir {get_direction(center, startPoint)};
    for (int i = 0; i < 8; i++) {

        Point point { next_point(center, curr_dir) };

        if (isZeroExamined == false) {
            if (point == Point(center.x, center.y+1) && get_pixel_value(img, Point(center.x, center.y+1)) == Non_edge_value) {
                isZeroExamined = true;
            }
        }

        if (get_pixel_value(img, point) != Non_edge_value) {
            return point;
        }

        Dir next_dir { cntrClockwiseNextDir(curr_dir) };
        curr_dir = next_dir;
    }
    return Non_point;
}

void my_findContour(const Mat& srcImage, Mat& dstImage, vector<Contour>& contours) {
    srcImage.copyTo(dstImage);
    int NBD { 1 };
    int LNBD { 1 };
    Point p1;
    Point p2;
    Point p3;
    Point p4;
    for (int i = 1; i < srcImage.rows; i++) {
        for (int j = 1; j < srcImage.cols; j++) {
            Point point(i,j);
            if(get_pixel_value(dstImage, point) != Non_edge_value) {
                Contour current_contour;

                // ===================================== STEP 1 =======================================
                if (isOuterBorder(dstImage, point)) {
                    NBD++;
                    current_contour.id = NBD;
                    current_contour.isOuterBorder = true;
                    p2 = { i, j-1 };
                } else if (isHoleBorder(dstImage, point)) {
                    NBD++;
                    current_contour.id = NBD;
                    current_contour.isOuterBorder = false;
                    p2 = { i, j+1 };
                    if (get_pixel_value(dstImage, point) > 1) {
                        LNBD = get_pixel_value(dstImage, point);
                    }
                } else {
                    // go to step 4
                    if (get_pixel_value(dstImage, point) != Edge_value) {
                        LNBD = abs(get_pixel_value(dstImage, point));
                    }
                    continue;
                }

                current_contour.points.push_back(point);

                // ===================================== STEP 2 =======================================


                if (LNBD == 1) { // frame
                    current_contour.parent = 1;
                } else {
                    if (contours.at(LNBD-2).isOuterBorder && current_contour.isOuterBorder) {
                        current_contour.parent = contours.at(LNBD-2).parent;
                    } else if (contours.at(LNBD-2).isOuterBorder && !current_contour.isOuterBorder) {
                        current_contour.parent = contours.at(LNBD-2).id;
                    } else if (!contours.at(LNBD-2).isOuterBorder && current_contour.isOuterBorder) {
                        current_contour.parent = contours.at(LNBD-2).id;
                    } else if (!contours.at(LNBD-2).isOuterBorder && !current_contour.isOuterBorder) {
                        current_contour.parent = contours.at(LNBD-2).parent;
                    }
                }


                // ===================================== STEP 3 =======================================
                // --------- STEP 3.1 ---------
                Point cs_point { clockwiseSearch(dstImage, point, p2) };
                if (cs_point != Non_point) {
                    p1 = cs_point;
                } else {
                    set_pixel_value(dstImage, point, -NBD);
                    // go to step 4
                    if (get_pixel_value(dstImage, point) != Edge_value) {
                        LNBD = abs(get_pixel_value(dstImage, point));
                    }
                    contours.push_back(current_contour);
                    continue;
                }

                // --------- STEP 3.2 ---------
                p2 = p1;
                p3 = point;

                while (true) {
                    bool isZeroExamined = false;
                    // --------- STEP 3.3 ---------
                    Point next_to_p2; // counter clockwise search start point

                    if (p3 == p2) {
                        break;
                    }

                    Dir dir_to_p2 { get_direction(p3, p2) };
                    Dir dir_to_p2_next { cntrClockwiseNextDir(dir_to_p2) };
                    next_to_p2 = next_point(p3, dir_to_p2_next);
                    Point ccs_point { cntrClockwiseSearch(dstImage, p3, next_to_p2, isZeroExamined) };
                    if (ccs_point != Non_point) {
                        p4 = ccs_point;
                    }

                    // --------- STEP 3.4 ---------
                    if (isZeroExamined == true) {
                        set_pixel_value(dstImage, p3, -NBD);
                    } else if (isZeroExamined == false && get_pixel_value(dstImage, p3) == Edge_value) {
                        set_pixel_value(dstImage, p3, NBD);
                    }
                    current_contour.points.push_back(p3);

                    // --------- STEP 3.5 ---------
                    if (p4 == point && p3 == p1) { // if return to starting point
                        break; // go to step 4
                    } else {
                        p2 = p3;
                        p3 = p4;
                        // go to step 2.3
                    }
                }

                // ===================================== STEP 4 =======================================
                if (get_pixel_value(dstImage, point) != Edge_value) {
                    LNBD = abs(get_pixel_value(dstImage, point));
                }

                contours.push_back(current_contour);
            }
        }
        LNBD = 1; // reset LNBD for new row
    }
}

void convert_to_binary(const Mat& srcImage, Mat& dstImage) {
    srcImage.copyTo(dstImage);
    for (int i = 0; i < srcImage.rows; i++) {
        for (int j = 0; j < srcImage.cols; j++) {
            if (get_pixel_value(srcImage, Point(i,j)) == -1) {
                set_pixel_value(dstImage, Point(i,j), 1);
            }
        }
    }
}

void swap_x_and_y(const Contour& contour, Contour& reversed_contour) {
    for (const Point& p : contour.points) {
            Point new_point;
            new_point.x = p.y;
            new_point.y = p.x;

            reversed_contour.points.push_back(new_point);
    }
}

void contour_threshold(const vector<vector<Point>>& srcContours, vector<vector<Point>>& dstContours, const int threshold) {
    for (const auto& contour : srcContours) {
        if (contourArea(contour) >= threshold){
            dstContours.push_back(contour);
        }
    }
}