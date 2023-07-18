#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("test9.jpg");

    if ( img.empty() )
    {
        cerr << "Could not open file" << endl;
        return ( 1 );
    }

    resize(img, img, Size(img.size().width / 4, img.size().height / 4), INTER_LINEAR);

    imshow("Scanner", img);
    waitKey(0);

    // Mat shift;
    // pyrMeanShiftFiltering(img, shift, 11, 21);

    // imshow("Scanner", shift);
    // waitKey(0);

    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    Mat otsu;
    threshold(img_gray, otsu, 0, 255, THRESH_OTSU+THRESH_BINARY);

    Mat kernel1;
    kernel1 = Mat::ones(Size(5, 5), CV_32F);
    dilate(otsu, otsu, kernel1, Point(-1, -1), 2);
    erode(otsu, otsu, kernel1, Point(-1, -1), 2);

    imshow("Scanner", otsu);
    waitKey(0);

    // Store the set of points in the image before assembling the bounding box
    std::vector<cv::Point> box_points;
    cv::Mat_<uchar>::iterator it = otsu.begin<uchar>();
    cv::Mat_<uchar>::iterator end = otsu.end<uchar>();
    for (; it != end; ++it)
    {
        if (*it) box_points.push_back(it.pos());
    }

    // Compute minimal bounding box
    cv::RotatedRect box = cv::minAreaRect(Mat(box_points));

    // Draw bounding box in the original image (debug purposes)
    cv::Point2f vertices[4];
    box.points(vertices);
    for (int i = 0; i < 4; ++i)
    {
            cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, 8);
    }

    imshow("Scanner", img);
    waitKey(0);

    vector<Point2f> pts(4);
    vector<Point2f> dst(4);

    pts[0] = vertices[1];
    pts[1] = vertices[2];
    pts[2] = vertices[3];
    pts[3] = vertices[0];

    dst[0].x = 0;
    dst[0].y = 0;
    dst[1].x = 40;
    dst[1].y = 0;
    dst[2].x = 40;
    dst[2].y = 1080;
    dst[3].x = 0;
    dst[3].y = 1080;

    Mat pTrans;
    pTrans = getPerspectiveTransform(pts, dst);

    Mat warped_img;
    warpPerspective(img, warped_img, pTrans, Size(40, 1080));

    imshow("Scanner", warped_img);
    waitKey(0);

    img_gray.setTo(Scalar(255), otsu);

    imshow("Scanner", img_gray);
    waitKey(0);

    double min, max;
    Point minLoc, maxLoc;

    minMaxLoc(img_gray, &min, &max, &minLoc, &maxLoc, img_gray);

    double alpha = 2;
    double beta = -255;
    Mat img_contrast;
    addWeighted(img_gray, alpha, img_gray, 0, beta, img_contrast);
    addWeighted(img_contrast, alpha, img_contrast, 0, beta, img_contrast);
    addWeighted(img_contrast, alpha, img_contrast, 0, beta, img_contrast);

    imshow("Scanner", img_contrast);
    waitKey(0);

    Mat img_contrast_thresh;
    threshold(img_contrast, img_contrast_thresh, 127, 255, THRESH_BINARY_INV);

    imshow("Scanner", img_contrast_thresh);
    waitKey(0);

    Mat kernel;
    Mat img_vert_mask;
    kernel = Mat::ones(Size(5, 5), CV_32F);
    erode(img_contrast_thresh, img_vert_mask, kernel);
    dilate(img_vert_mask, img_vert_mask, kernel, Point(-1, -1), 1);

    imshow("Scanner", img_vert_mask);
    waitKey(0);

    vector<vector<Point> > contours;
    findContours(img_vert_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat img_cont;

    img.copyTo(img_cont);

    if ( img_cont.empty() )
    {
        cerr << "Could not open file" << endl;
        return ( 1 );
    }

    drawContours(img_cont, contours, -1, Scalar(0, 0, 255));

    imshow("Scanner", img_cont);
    waitKey(0);

    Mat final = Mat::zeros(img.size(), CV_32F);
    Mat mask = Mat::zeros(img_gray.size(), CV_8U);
    Mat palette;

    // Mat pixels;
    // img.reshape(3, 0).convertTo(pixels, CV_32F);

    // int n_colors = 4;
    // TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 200, 0.1);
    // int flags = KMEANS_RANDOM_CENTERS;

    // Mat labels;

    // kmeans(pixels, n_colors, labels, criteria, 10, flags, palette);

    // cout << img << endl;
    // cout << pixels.size() << endl;
    // cout << palette.size() << endl;

    // if (contours.size() > 17) {
    //     cerr << "too many contours" << endl;
    //     return ( 1 );
    // }

    vector<Point> points(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        mask.setTo(Scalar(0));
        drawContours(mask, contours, i, 255, -1);
        drawContours(final, contours, i, mean(img, mask), -1);
        Moments M = moments(contours[i]);
        int cX = static_cast<int>(M.m10 / M.m00);
        int cY = static_cast<int>(M.m01 / M.m00);

        if (cY < img.size().height && cY > 0 && cX < img.size().width && cX > 0) {
            points[i] = Point(cX, cY);

            Mat dominant = Mat(img.size(), CV_8UC3, mean(img, mask));

            cout << cX << " " << cY << endl;

            circle(img, Point(cX, cY), 3, Scalar(255, 0, 0), -1);
        }

        // imshow("Scanner", mask);
        // waitKey(0);

        // imshow("Scanner 2", dominant);
        // waitKey(0);
    }

    // size_t pointIdxs[points.size()];

    // if (points.size() > 5) {
    //     first_combination(pointIdxs, 6);

    //     for (int i = 0; i < 6; i++)
    //     {
    //         for (int j = 0; j < 6; j++)
    //         {
    //             if (i != j)
    //             {

    //             }
    //         }
    //     }
    // }

    Vec4f lp;

    fitLine(points, lp, DIST_L2, 0, 0.01, 0.01);

    cout << lp << endl;


    double t0 = (0- lp[3])/lp[1];
    double t1 = (img.size().height-lp[3])/lp[1];

    Point p0(lp[2] + (t0 * lp[0]), lp[3] + (t0 * lp[1]));
    Point p1(lp[2] + (t1 * lp[0]), lp[3] + (t1 * lp[1]));

    cout << p0 << p1 << endl;

    line(img, p0, p1, Scalar(0, 255, 0), 2, LINE_8);

    cout << "line drawn" << endl;

    imshow("Scanner", img);
    waitKey(0);

    return ( 0 );
}