#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

bool findBoxFromContours(vector<vector<Point>> contours, Point2f *vertices){
    
    list<RotatedRect> boxes;
    
    for(vector<Point> contour : contours)
    {
        // Compute minimal bounding box
        cv::RotatedRect box = cv::minAreaRect(Mat(contour));

        double exp = 0.037037;

        double act = box.size.aspectRatio();

        if ((2 * min(exp, act)) / (exp + act) > 0.90)
        {
            boxes.push_back(box);
        }
    }

    if (boxes.size() == 0) {
        return false;
    }

    RotatedRect biggest_box = boxes.front();

    boxes.pop_front();

    for (RotatedRect curr_box : boxes) {
        if (curr_box.size.area() > biggest_box.size.area()) {
            biggest_box = curr_box;
        }
    }

    biggest_box.points(vertices);

    return true;
}

int main()
{
    Mat img = imread("test2.jpg");

    if ( img.empty() )
    {
        cerr << "Could not open file" << endl;
        return ( 1 );
    }

    resize(img, img, Size(img.size().width / 4, img.size().height / 4), INTER_LINEAR);

    namedWindow("Scanner", WINDOW_AUTOSIZE);

    // Mat shift;
    // pyrMeanShiftFiltering(img, shift, 11, 21);

    // imshow("Scanner", shift);
    // waitKey(0);

    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    imshow("Scanner", img_gray);
    waitKey(0);

    Mat img_filtered;
    bilateralFilter(img_gray, img_filtered, 5, 40, 40);

    GaussianBlur(img_filtered, img_filtered, Size(5, 5), 0);

    imshow("Scanner", img_filtered);
    waitKey(0);

    Mat canny;
    Canny(img_filtered, canny, 100, 200);

    imshow("Scanner", canny);
    waitKey(0);

    vector<vector<Point>> contours;
    findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Point2f vertices[4];    

    if (!findBoxFromContours(contours, vertices))
    {
        cerr << "Could not find box" << endl;
        return ( 1 );
    }

    Mat img_box = img.clone();
    for (int i = 0; i < 4; ++i)
    {
            cv::line(img_box, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, 8);
    }

    imshow("Scanner", img_box);
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

    Mat shift;
    pyrMeanShiftFiltering(warped_img, shift, 11, 21);

    vector<int> centerpoints{
        28,
        88,
        148,
        208,
        268,
        328,
        388,
        448,
        508,
        568,
        628,
        688,
        748,
        808,
        868,
        928
    };

    Mat points_img = shift.clone();

    for (size_t i = 0; i < 16; i++)
    {

        cout << "iter " << i << endl;

        circle(points_img, Point(20, centerpoints[i]), 3, Scalar(255, 0, 0), -1);

        Mat labels;

        TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 200, 0.1);
        int flags = KMEANS_PP_CENTERS;

        Mat mask = Mat::zeros(1080, 40, CV_8U);

        rectangle(mask, Rect(Point(5, centerpoints[i] - 15), Point(35, centerpoints[i] + 15)), Scalar(255), -1);

        Mat centers, data;
        shift.convertTo(data, CV_32F);    
        // reshape into 3 columns (one per channel, in BGR order) and as many rows as the total number of pixels in img
        data = data.reshape(1, data.total()); 

        int nbWhitePixels = cv::countNonZero(mask);
        cv::Mat dataMasked = cv::Mat(nbWhitePixels, 3, CV_32F, cv::Scalar(0));
        cv::Mat maskFlatten = mask.reshape(1, mask.total());
            
        // filter data by the mask
        int idx = 0;
        for (int k = 0; k < mask.total(); k++)
        {
            int val = maskFlatten.at<uchar>(k, 0);          
            if (val != 0)
            {
                float val0 = data.at<float>(k, 0);
                float val1 = data.at<float>(k, 1);
                float val2 = data.at<float>(k, 2);
                dataMasked.at<float>(idx,0) = val0;
                dataMasked.at<float>(idx,1) = val1;
                dataMasked.at<float>(idx,2) = val2;
                idx++;
            }
        }

        // apply k-means    
        cv::kmeans(dataMasked, 2, labels, criteria, 10, flags, centers);

        vector<int> args(2);

        for (int j = 0; j < labels.size().height; j++) {
            args[labels.at<int>(j)]++;
        }

        // reshape to a single column of Vec3f pixels
        centers = centers.reshape(3, centers.rows);  
        dataMasked = dataMasked.reshape(3, dataMasked.rows);
        data = data.reshape(3, data.rows);

        if (args[0] > args[1]) {
            rectangle(shift, Rect(Point(0, centerpoints[i] - 20), Point(40, centerpoints[i] + 20)), Scalar(centers.at<Vec3f>(0)[0], centers.at<Vec3f>(0)[1], centers.at<Vec3f>(0)[2]), -1);
        } else {
            rectangle(shift, Rect(Point(0, centerpoints[i] - 20), Point(40, centerpoints[i] + 20)), Scalar(centers.at<Vec3f>(1)[0], centers.at<Vec3f>(1)[1], centers.at<Vec3f>(1)[2]), -1);
        }


        imshow("Scanner", shift);
        waitKey(0);

    }

    resize(points_img, points_img, Size(30, 810));

    resize(shift, shift, Size(30, 810));

    imshow("Scanner", points_img);
    waitKey(0);

    imshow("Scanner", shift);
    waitKey(0);

    // img_gray.setTo(Scalar(255), otsu);

    // imshow("Scanner", img_gray);
    // waitKey(0);

    // double min, max;
    // Point minLoc, maxLoc;

    // minMaxLoc(img_gray, &min, &max, &minLoc, &maxLoc, img_gray);

    // double alpha = 2;
    // double beta = -255;
    // Mat img_contrast;
    // addWeighted(img_gray, alpha, img_gray, 0, beta, img_contrast);
    // addWeighted(img_contrast, alpha, img_contrast, 0, beta, img_contrast);
    // addWeighted(img_contrast, alpha, img_contrast, 0, beta, img_contrast);

    // imshow("Scanner", img_contrast);
    // waitKey(0);

    // Mat img_contrast_thresh;
    // threshold(img_contrast, img_contrast_thresh, 127, 255, THRESH_BINARY_INV);

    // imshow("Scanner", img_contrast_thresh);
    // waitKey(0);

    // Mat kernel;
    // Mat img_vert_mask;
    // kernel = Mat::ones(Size(5, 5), CV_32F);
    // erode(img_contrast_thresh, img_vert_mask, kernel);
    // dilate(img_vert_mask, img_vert_mask, kernel, Point(-1, -1), 1);

    // imshow("Scanner", img_vert_mask);
    // waitKey(0);

    // vector<vector<Point> > contours;
    // findContours(img_vert_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Mat img_cont;

    // img.copyTo(img_cont);

    // if ( img_cont.empty() )
    // {
    //     cerr << "Could not open file" << endl;
    //     return ( 1 );
    // }

    // drawContours(img_cont, contours, -1, Scalar(0, 0, 255));

    // imshow("Scanner", img_cont);
    // waitKey(0);

    // Mat final = Mat::zeros(img.size(), CV_32F);
    // Mat mask = Mat::zeros(img_gray.size(), CV_8U);
    // Mat palette;

    // // Mat pixels;
    // // img.reshape(3, 0).convertTo(pixels, CV_32F);

    // // int n_colors = 4;
    // // TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 200, 0.1);
    // // int flags = KMEANS_RANDOM_CENTERS;

    // // Mat labels;

    // // kmeans(pixels, n_colors, labels, criteria, 10, flags, palette);

    // // cout << img << endl;
    // // cout << pixels.size() << endl;
    // // cout << palette.size() << endl;

    // // if (contours.size() > 17) {
    // //     cerr << "too many contours" << endl;
    // //     return ( 1 );
    // // }

    // vector<Point> points(contours.size());

    // for (size_t i = 0; i < contours.size(); i++)
    // {
    //     mask.setTo(Scalar(0));
    //     drawContours(mask, contours, i, 255, -1);
    //     drawContours(final, contours, i, mean(img, mask), -1);
    //     Moments M = moments(contours[i]);
    //     int cX = static_cast<int>(M.m10 / M.m00);
    //     int cY = static_cast<int>(M.m01 / M.m00);

    //     if (cY < img.size().height && cY > 0 && cX < img.size().width && cX > 0) {
    //         points[i] = Point(cX, cY);

    //         Mat dominant = Mat(img.size(), CV_8UC3, mean(img, mask));

    //         cout << cX << " " << cY << endl;

    //         circle(img, Point(cX, cY), 3, Scalar(255, 0, 0), -1);
    //     }

    //     // imshow("Scanner", mask);
    //     // waitKey(0);

    //     // imshow("Scanner 2", dominant);
    //     // waitKey(0);
    // }

    // // size_t pointIdxs[points.size()];

    // // if (points.size() > 5) {
    // //     first_combination(pointIdxs, 6);

    // //     for (int i = 0; i < 6; i++)
    // //     {
    // //         for (int j = 0; j < 6; j++)
    // //         {
    // //             if (i != j)
    // //             {

    // //             }
    // //         }
    // //     }
    // // }

    // Vec4f lp;

    // fitLine(points, lp, DIST_L2, 0, 0.01, 0.01);

    // cout << lp << endl;


    // double t0 = (0- lp[3])/lp[1];
    // double t1 = (img.size().height-lp[3])/lp[1];

    // Point p0(lp[2] + (t0 * lp[0]), lp[3] + (t0 * lp[1]));
    // Point p1(lp[2] + (t1 * lp[0]), lp[3] + (t1 * lp[1]));

    // cout << p0 << p1 << endl;

    // line(img, p0, p1, Scalar(0, 255, 0), 2, LINE_8);

    // cout << "line drawn" << endl;

    // imshow("Scanner", img);
    // waitKey(0);

    destroyAllWindows();

    return ( 0 );
}