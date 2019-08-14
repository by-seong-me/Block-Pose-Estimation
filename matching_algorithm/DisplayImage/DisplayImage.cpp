#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


cv::RotatedRect getBoundingRectPCA( cv::Mat& binaryImg );

/** @function main */
int main( int argc, char** argv )
{   
  /// Load source image and convert it to gray
  Mat src = imread( argv[1], 0 );

  namedWindow( "original image", CV_WINDOW_AUTOSIZE );
  imshow("original image", src);
  
  cv::RotatedRect boundRect = getBoundingRectPCA( src );
    
  Mat drawing = src.clone();
  // contour
    // rotated rectangle
  Point2f rect_points[4]; boundRect.points( rect_points );
  for( int j = 0; j < 4; j++ )
    line( drawing, rect_points[j], rect_points[(j+1)%4], 255, 1, 8 );
  
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
  waitKey(0);

  return(0);
}


cv::RotatedRect getBoundingRectPCA( cv::Mat& binaryImg ) {
cv::RotatedRect result;

//1. convert to matrix that contains point coordinates as column vectors
int count = cv::countNonZero(binaryImg);
if (count == 0) {
    std::cout << "getBoundingRectPCA() encountered 0 pixels in binary image!" << std::endl;
    return cv::RotatedRect();
}

cv::Mat data(2, count, CV_32FC1);
int dataColumnIndex = 0;
for (int row = 0; row < binaryImg.rows; row++) {
    for (int col = 0; col < binaryImg.cols; col++) {
        if (binaryImg.at<unsigned char>(row, col) != 0) {
            data.at<float>(0, dataColumnIndex) = (float) col; //x coordinate
            data.at<float>(1, dataColumnIndex) = (float) (binaryImg.rows - row); //y coordinate, such that y axis goes up
            ++dataColumnIndex;
        }
    }
}

//2. perform PCA
const int maxComponents = 1;
cv::PCA pca(data, cv::Mat() /*mean*/, CV_PCA_DATA_AS_COL, maxComponents);
//result is contained in pca.eigenvectors (as row vectors)
//std::cout << pca.eigenvectors << std::endl;

//3. get angle of principal axis
float dx = pca.eigenvectors.at<float>(0, 0);
float dy = pca.eigenvectors.at<float>(0, 1);
float angle = atan2f(dy, dx)  / (float)CV_PI*180.0f;

//find the bounding rectangle with the given angle, by rotating the contour around the mean so that it is up-right
//easily finding the bounding box then
cv::Point2f center(pca.mean.at<float>(0,0), binaryImg.rows - pca.mean.at<float>(1,0));
cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, -angle, 1);
cv::Mat rotationMatrixInverse = cv::getRotationMatrix2D(center, angle, 1);

std::vector<std::vector<cv::Point> > contours;
vector<Vec4i> hierarchy;
cv::findContours(binaryImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

vector<Point> contour;
if (contours.size() != 1) {

    int max_num = 0;
    int max_index = 0;

    for(int i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() > max_num)
           { max_num = contours[i].size(); max_index = i;}
    }
    contour = contours[max_index];
}

//turn vector of points into matrix (with points as column vectors, with a 3rd row full of 1's, i.e. points are converted to extended coords)
cv::Mat contourMat(3, contour.size(), CV_64FC1);
double* row0 = contourMat.ptr<double>(0);
double* row1 = contourMat.ptr<double>(1);
double* row2 = contourMat.ptr<double>(2);
for (int i = 0; i < (int) contours[0].size(); i++) {
    row0[i] = (double) (contours[0])[i].x;
    row1[i] = (double) (contours[0])[i].y;
    row2[i] = 1;
}

cv::Mat uprightContour = rotationMatrix*contourMat;

//get min/max in order to determine width and height
double minX, minY, maxX, maxY;
cv::minMaxLoc(cv::Mat(uprightContour, cv::Rect(0, 0, contours[0].size(), 1)), &minX, &maxX); //get minimum/maximum of first row
cv::minMaxLoc(cv::Mat(uprightContour, cv::Rect(0, 1, contours[0].size(), 1)), &minY, &maxY); //get minimum/maximum of second row

int minXi = cvFloor(minX);
int minYi = cvFloor(minY);
int maxXi = cvCeil(maxX);
int maxYi = cvCeil(maxY);

//fill result
result.angle = angle;
result.size.width = (float) 25;
result.size.height = (float) 50;

//Find the correct center:
cv::Mat correctCenterUpright(3, 1, CV_64FC1);
correctCenterUpright.at<double>(0, 0) = maxX - result.size.width/2;
correctCenterUpright.at<double>(1,0) = maxY - result.size.height/2;
correctCenterUpright.at<double>(2,0) = 1;
cv::Mat correctCenterMat = rotationMatrixInverse*correctCenterUpright;
cv::Point correctCenter = cv::Point(cvRound(correctCenterMat.at<double>(0,0)), cvRound(correctCenterMat.at<double>(1,0)));

result.center = correctCenter;

return result;
}