#include "my_minarearect.h"
#include "myconvexhull.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types.hpp>

#include <utility>
#include <limits>
#include <cmath>

using namespace std;
using namespace cv;

struct Rectangle {
    vector<Point> points;
    float angle;
};

struct Pair {
    Point point {};
    int index {};

    Pair() = default;
    Pair(Point p, int i) {
        point = p;
        index = i;
    }
};

Pair leftmost_point(const vector<Point>& points) {
    Pair pr(points.at(0), 0);

    for (int i = 1; i < points.size(); i++) {
        Point temp { points.at(i) };
        if (temp.x < pr.point.x) {
            pr.point = temp;
            pr.index = i;
        } else if (temp.x == pr.point.x && temp.y < pr.point.y) {
            pr.point = temp;
            pr.index = i;
        }
    }

    return pr;
}

Pair uppermost_point(const vector<Point>& points) {
    Pair pr(points.at(0), 0);

    for (int i = 1; i < points.size(); i++) {
        Point temp { points.at(i) };
        if (temp.y < pr.point.y) {
            pr.point = temp;
            pr.index = i;
        } else if (temp.y == pr.point.y && temp.x > pr.point.x) {
            pr.point = temp;
            pr.index = i;
        }
    }

    return pr;
}

Pair rightmost_point(const vector<Point>& points) {
    Pair pr(points.at(0), 0);

    for (int i = 1; i < points.size(); i++) {
        Point temp { points.at(i) };
        if (temp.x > pr.point.x) {
           pr.point = temp;
           pr.index = i;
        } else if (temp.x == pr.point.x && temp.y > pr.point.y) {
            pr.point = temp;
            pr.index = i;
        }
    }

    return pr;
}

Pair lowermost_point(const vector<Point>& points) {
    Pair pr(points.at(0), 0);

    for (int i = 1; i < points.size(); i++) {
        Point temp { points.at(i) };
        if (temp.y > pr.point.y) {
            pr.point = temp;
            pr.index = i;
        } else if (temp.y == pr.point.y && temp.x < pr.point.x) {
            pr.point = temp;
            pr.index = i;
        }
    }

    return pr;
}

vector<Pair> find_end_points_with_index(const vector<Point>& points) {
    vector<Pair> endPoints;
    endPoints.push_back(leftmost_point(points));
    endPoints.push_back(uppermost_point(points));
    endPoints.push_back(rightmost_point(points));
    endPoints.push_back(lowermost_point(points));

    return endPoints;
}

float min(const float a, const float b, const float c, const float d) {
    return min(std::min(a, b), std::min(c, d));
}

float min(const vector<double>& angles) {
    return min(std::min(angles.at(0), angles.at(1)), std::min(angles.at(2), angles.at(3)));
}

float calculate_angle(const Point a, const Point b) {
    constexpr float eps { std::numeric_limits<float>::epsilon() };

    int x_diff { b.x - a.x };
    int y_diff { b.y - a.y };

    // atan2 ?
    if ((x_diff > 0 && y_diff <= 0) || (x_diff < 0 && y_diff >= 0)) {
        return atan((float)(abs(y_diff)) / ((float)(abs(x_diff)) + eps)) * 180 / M_PI;
    } else if ((x_diff >= 0 && y_diff > 0) || (x_diff <= 0 && y_diff < 0)) {
        return atan((float)(abs(x_diff)) / ((float)(abs(y_diff)) + eps)) * 180 / M_PI;
    }
    cerr << "Angle can not be calculated!\n";
    abort();
}

Point rotate_point(const double angle, const Point& point, const Point& center) {
    double angleInRads { angle * M_PI / 180 };
    double s { sin(angleInRads) };
    double c { cos(angleInRads) };

    Point newPoint;

    newPoint.x = (point.x - center.x) * c - (point.y - center.y) * s + center.x;
    newPoint.y = (point.x - center.x) * s + (point.y - center.y) * c + center.y;

    return newPoint;
}

void rotate_shape(const vector<Point> points, vector<Point>& rotated_points, const double angle, const Point& center) {
    for (const Point& p : points) {
        Point rotated_point { rotate_point(angle, p, center) };
        rotated_points.push_back(rotated_point);
    }
}

Point find_center(const vector<Point>& points) {
    int new_x { 0 };
    int new_y { 0 };

    for (const Point& p : points) {
        new_x += p.x;
        new_y += p.y;
    }

    new_x /= points.size();
    new_y /= points.size();

    return Point(new_x, new_y);
}

void find_rectangle_points(const vector<Point>& end_points, vector<Point>& points) {
    int min_x { end_points.at(0).x };
    int min_y{ end_points.at(0).y };
    int max_x { end_points.at(0).x };
    int max_y{ end_points.at(0).y };
    for (int i = 1; i < end_points.size(); i++) {
        if (end_points.at(i).x < min_x) {
            min_x = end_points.at(i).x;
        }

        if (end_points.at(i).y < min_y) {
            min_y = end_points.at(i).y;
        }

        if (end_points.at(i).x > max_x) {
            max_x = end_points.at(i).x;
        }

        if (end_points.at(i).y > max_y) {
            max_y = end_points.at(i).y;
        }
    }

    Point up_left { min_x, min_y };
    Point up_right { max_x, min_y };
    Point down_right { max_x, max_y };
    Point down_left { min_x, max_y };

    points.push_back(up_left);
    points.push_back(up_right);
    points.push_back(down_right);
    points.push_back(down_left);

}

int find_rectangle_area(const vector<Point>& points) {
    int x_lenght {points.at(2).x - points.at(0).x};
    int y_lenght {points.at(2).y - points.at(0).y};

    return x_lenght * y_lenght;
}

void shift_points(const vector<Point>& points, vector<Point>& shifted_points, const int x_shift, const int y_shift) {
    shifted_points = points;

    for (Point& point : shifted_points) {
        point.x += x_shift;
        point.y += y_shift;
    }
}


double calculate_polygon_area(const vector<Point>& points) {
    double area { 0 };

    for (int i = 0; i < points.size(); i++) {
        Point current_point { points.at(i) };
        Point next_point { points.at((i + 1) % points.size()) };

        area += (current_point.x + next_point.x) * (next_point.y - current_point.y);
    }

    return area / 2;
}

Point find_polygon_center(const vector<Point>& points) {
    int x { 0 };
    int y { 0 };
    double area { calculate_polygon_area(points) };

    for (int i = 0; i < points.size(); i++) {
        Point current_point { points.at(i) };
        Point next_point { points.at((i + 1) % points.size()) };

        x += (current_point.x + next_point.x) * (current_point.x * next_point.y - next_point.x * current_point.y);
        y += (current_point.y + next_point.y) * (current_point.x * next_point.y - next_point.x * current_point.y);

    }

    x /= 6 * area;
    y /= 6 * area;

    return Point(x, y);
}



void my_minAreaRect(const vector<Point>& convex_hull, vector<Point>& minAreaRect) {
    vector<Point> rotated_shape { convex_hull };
    Point center { find_polygon_center(convex_hull) };

    vector<vector<Point>> rectangle_history;
    vector<vector<Point>> shape_history;
    vector<double> rotation_history;
    vector<int> area_history;

    vector<Point> initial_rectangle;
    find_rectangle_points(convex_hull, initial_rectangle);

    rectangle_history.push_back(initial_rectangle);
    rotation_history.push_back(0);
    area_history.push_back(find_rectangle_area(initial_rectangle));
    shape_history.push_back(rotated_shape);

    double total_rotation { 0 };

    while (true) {
        vector<Pair> endpoints { find_end_points_with_index(rotated_shape) };

        // calculate angles
        vector<double> angles;
        for (int i = 0; i < 4; i++) {
            Point p { endpoints.at(i).point };
            Point p1 { rotated_shape.at((endpoints.at(i).index  + 1) % 4)};

            angles.push_back(calculate_angle(p, p1));
        }

        double theta { 9999 };
        for (double angle : angles) {
            if (angle < theta && angle != 0) {
                theta = angle;
            }
        }

        total_rotation += theta;

        // rotate shape
        vector<Point> temp_shape;
        rotate_shape(rotated_shape, temp_shape, theta, center);
        rotated_shape = temp_shape;

        vector<Point> rect;
        find_rectangle_points(rotated_shape, rect);

        int area { find_rectangle_area(rect) };

        area_history.push_back(area);
        rectangle_history.push_back(rect);
        rotation_history.push_back(total_rotation);
        shape_history.push_back(rotated_shape);

        if (total_rotation > 90) {
            break;
        }
    }


    // find minimum area
    int min_area { area_history.at(0) };
    int min_index { 0 };
    for (int i = 1; i < area_history.size(); i++) {
        if (area_history.at(i) < min_area) {
            min_area = area_history.at(i);
            min_index = i;
        }
    }

    // rotate min rectangle and its rotated shape
    double rotation { rotation_history.at(min_index) };
    vector<Point> min_rect_shape_rotated { shape_history.at(min_index) };
    vector<Point> min_rectangle_not_rotated { rectangle_history.at(min_index) };
    vector<Point> min_rect_shape_rerotated;
    vector<Point> min_rectangle_rotated;
    rotate_shape(min_rect_shape_rotated, min_rect_shape_rerotated, -rotation, center);
    rotate_shape(min_rectangle_not_rotated, min_rectangle_rotated, -rotation, center);


    // shift minimum rectangle
    Point center_to_shift { find_polygon_center(min_rect_shape_rerotated) };
    int x_diff { center.x - center_to_shift.x };
    int y_diff { center.y - center_to_shift.y };
    vector<Point> shifted_shape;
    vector<Point> shifted_rectangle;
    shift_points(min_rect_shape_rerotated, shifted_shape, x_diff, y_diff);
    shift_points(min_rectangle_rotated, shifted_rectangle, x_diff, y_diff);

    minAreaRect = shifted_rectangle;

}




