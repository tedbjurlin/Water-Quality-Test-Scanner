#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

static vector<cv::Point> detect_edges( Mat& image);

static double get_cosine_angle_between_vectors( cv::Point pt1, cv::Point pt2, cv::Point pt0 );

static vector<vector<cv::Point> > find_squares(Mat& image);

static float get_width(vector<cv::Point>& square);

static float get_height(vector<cv::Point>& square);