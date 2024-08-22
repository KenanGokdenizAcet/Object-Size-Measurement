#include "myconvexhull.h"
#include "my_minarearect.h"
#include "mycontour.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

#define TEXT_WEIGHT 1.5
#define TEXT_SIZE 0.4
#define FONT FONT_HERSHEY_SIMPLEX

namespace Colors{
    const cv::Scalar    Green{50, 255, 100},
                        Red{0, 0, 255},
                        Blue{155, 100,0},
                        Yellow{0,255,255},
                        Orange{0,155,255},
                        Purple{255,0,255};
}

void draw_shape(Mat& image, const vector<Point>& points, const Scalar color, int thickness = 1) {
    for (int i = 0; i < points.size(); i++) {
        line(image, points.at(i), points.at((i + 1) % points.size()), color, thickness);
    }
}

Point find_line_center(const Point& p1, const Point& p2) {
    Point center;
    center.x = (p1.x + p2.x) / 2;
    center.y = (p1.y + p2.y) / 2;
    return center;
}

string inline double_to_string(const double value, const int x) {
    return to_string(value).substr(0, to_string(value).find(".") + x);
}

double measure_distance(Point p1, Point p2) {
    int x_lenght { p1.x - p2.x };
    int y_lenght { p1.y - p2.y };

    return sqrt((pow(x_lenght, 2) + pow(y_lenght, 2)));
}

double measure_real_distance(const Point& p1, const Point& p2, const double x_ratio, const double y_ratio) {
    int x_lenght { p1.x - p2.x };
    int y_lenght { p1.y - p2.y };
    double true_x_lenght {x_lenght * x_ratio};
    double true_y_lenght {y_lenght * y_ratio};

    return sqrt((pow(true_x_lenght, 2) + pow(true_y_lenght, 2)));
}

void draw_rectangle_with_length(Mat& img, const vector<Point>& rec, const double x_ratio, const double y_ratio) {
    vector<Point> line_centers;

    for (int p = 0; p < 4; p++) {
        line(img, rec.at(p), rec.at((p+1) % 4), Colors::Yellow, 2);
        line_centers.push_back(find_line_center(rec.at(p), rec.at((p+1)%4)));
    }

    int left_line_center_index = 0;
    for (int i = 1; i < 4; i++) {
        if (line_centers.at(i).x < line_centers.at(left_line_center_index).x) {
            left_line_center_index = i;
        }
    }

    int up_line_center_index = 0;
    for(int i = 1; i < 4; i++) {
        if(line_centers.at(i).y < line_centers.at(up_line_center_index).y) {
            up_line_center_index = i;
        }
    }

    Point left_text_point { line_centers.at(left_line_center_index) };
    left_text_point.x -= 25;

    Point up_text_point { line_centers.at(up_line_center_index) };
    up_text_point.x -= 10;
    up_text_point.y -= 5;

    double left_lenght = measure_real_distance(rec.at(left_line_center_index), rec.at((left_line_center_index + 1)%4), x_ratio, y_ratio);
    double up_lenght = measure_real_distance(rec.at(up_line_center_index), rec.at((up_line_center_index + 1)%4), x_ratio, y_ratio);

    putText(img, double_to_string(left_lenght, 2), left_text_point, FONT, TEXT_SIZE, Colors::Red, TEXT_WEIGHT);
    putText(img, double_to_string(up_lenght, 2), up_text_point, FONT, TEXT_SIZE, Colors::Red, TEXT_WEIGHT);
}

void draw_line_with_distance(Mat& output, Point p1, Point p2, double x_ratio, double y_ratio) {
    Point line_center { find_line_center(p1, p2) };
    double distance { measure_real_distance(p1, p2, x_ratio, y_ratio) };

    line(output, p1, p2, Colors::Green);
    putText(output, double_to_string(distance, 2), line_center, FONT, TEXT_SIZE, Colors::Red, TEXT_WEIGHT);
}



int main() {

    Mat img { imread("test_image") };
    Mat processed_img, edge;
    cvtColor(img, processed_img, COLOR_BGR2GRAY);
    GaussianBlur(processed_img, processed_img, Size(5,5), 1);
    Canny(processed_img, edge, 150, 200);
    convert_to_binary(edge, edge);

    vector<Contour> contours;
    Mat image_with_contour;
    my_findContour(edge, image_with_contour,contours);

    // find contours larger than threshold
    vector<Contour> th_contours;
    for (const auto& contour : contours) {
        if (contour.parent == 1 && contour.points.size() > 100) {
            th_contours.push_back(contour);
        }
    }

    Contour new_contour;
    swap_x_and_y(th_contours.at(3), new_contour);
    vector<Point> convex_points;

    vector<Point> ref_rect;

    my_convex_hull(new_contour.points, convex_points);
    my_minAreaRect(convex_points, ref_rect);
    draw_shape(img, convex_points, Colors::Purple, 1);

    double true_width = 0.955; // inch
    double true_height = 0.955; // inch

    double width { measure_distance(ref_rect.at(0), ref_rect.at(1)) };
    double height { measure_distance(ref_rect.at(1), ref_rect.at(2)) };

    double x_ratio {true_width / width};
    double y_ratio {true_height / height};

    vector<vector<Point>> rectangles;

    for (int i = 0; i < th_contours.size(); i++) {
        Contour new_contour;
        swap_x_and_y(th_contours.at(i), new_contour);

        vector<Point> convex_points;
        vector<Point> min_area_rect;

        my_convex_hull(new_contour.points, convex_points);

        my_minAreaRect(convex_points, min_area_rect);

        rectangles.push_back(min_area_rect);

        draw_shape(img, min_area_rect, Colors::Orange, 1);
        draw_shape(img, convex_points, Colors::Blue, 1);

        draw_rectangle_with_length(img, min_area_rect, x_ratio, y_ratio);

    }

    imshow("image", img);

    waitKey(0);

    return 0;
}