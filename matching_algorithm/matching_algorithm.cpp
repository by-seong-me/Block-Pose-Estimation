#include <math.h>       /* atan */
#include <opencv2/viz.hpp>

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

//#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "matching_algorithm.h"

#define PI 3.14159265

using namespace cv;
using namespace std;

Rect lineFitting(cv::Mat image);
double GetDist(Plane::Plane_model plane, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outlier);
cv::Mat DistantPlaneSegmentation(Plane::Plane_model plane, cv::Mat red_pcloud, float distance);
cv::Mat dominant_plane_projection(Plane::Plane_model plane, cv::Mat pCloud);
cv::Mat get_projected_image(Plane::Plane_model plane, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);
cv::Mat RectFitting(cv::Mat projected_image, std::string string);
void LetsFindBox(cv::Mat projected_image);
void FindBoundingBox(cv::Mat image_RGB, cv::Mat image_depth);
void CloudViewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2);
void Merge(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> color_vector);
void vizPointCloud(cv::Mat image_RGB, pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud);
void GetPose(std::string color_string, pcl::PointCloud<pcl::PointXYZ> color_synthetic);
pcl::PointXYZ GetUnitY(pcl::PointXYZ origin, pcl::PointXYZ unit_x, Plane::Plane_model plane);
pcl::PointCloud<pcl::PointXYZ> makeSyntheticCloud(std::string color_string);
pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_dominant_plane_projection(Plane::Plane_model plane, cv::Mat pCloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr back_projection(cv::Mat image, std::string string);
pcl::PointCloud<pcl::PointXYZ>::Ptr makeForm (pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr cv2pcl(cv::Mat image);

bool colorCheck(cv::Mat color_image, cv::Mat& color_pCloud, cv::Mat pCloud);

double getDistant(pcl::PointXYZ pt_1, pcl::PointXYZ pt_2);
int getStep(float number);

void BoundingBox(Plane::Plane_model dominant_plane, cv::Mat pCloud_outlier, std::string color_string);
Point Rotate(float x, float y, float xm, float ym, float theta);

pcl::PointCloud<pcl::PointXYZ> GetRedSynthetic();
pcl::PointCloud<pcl::PointXYZ> GetOrangeSynthetic();
pcl::PointCloud<pcl::PointXYZ> GetYellowSynthetic();
pcl::PointCloud<pcl::PointXYZ> GetGreenSynthetic();
pcl::PointCloud<pcl::PointXYZ> GetBlueSynthetic();
pcl::PointCloud<pcl::PointXYZ> GetPurpleSynthetic();
pcl::PointCloud<pcl::PointXYZ> GetBrownSynthetic();

cv::Scalar fitEllipseColor = Scalar(255,  0,  0);
cv::Mat processImage(cv::Mat image, std::string string);
int sliderPos = 70;

int number;

cv::Mat image_RGB;
cv::Mat image_depth;

cv::Mat Ineedoutlier;

pcl::PointXYZ origin;
pcl::PointXYZ unit_x;
pcl::PointXYZ unit_y;

std::vector<pcl::PointXYZ> keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<pcl::PointXYZ> red_keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<pcl::PointXYZ> orange_keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<pcl::PointXYZ> yellow_keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<pcl::PointXYZ> green_keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<pcl::PointXYZ> blue_keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<pcl::PointXYZ> purple_keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<pcl::PointXYZ> brown_keypoints = std::vector<pcl::PointXYZ> (4);

cv::Point rect_points[4];
std::vector<Point2f> rectpoints = std::vector<Point2f> (4);

std::vector<float> unit_vector = std::vector<float> (3);

float a;
float b;
float c;
float d;

bool minus_x;
bool minus_y; 

int dim_w;
int dim_h;

int min_x = 0;
int min_y = 0;

double block_height;

double red_block_height;
double orange_block_height;
double yellow_block_height;
double green_block_height;
double blue_block_height;
double purple_block_height;
double brown_block_height;

cv::Mat red_pCloud;
cv::Mat orange_pCloud;
cv::Mat yellow_pCloud;
cv::Mat green_pCloud;
cv::Mat blue_pCloud;
cv::Mat purple_pCloud;
cv::Mat brown_pCloud;

bool bool_red = false;
bool bool_orange = false;
bool bool_yellow = false;
bool bool_green = false;
bool bool_blue = false;
bool bool_purple = false;
bool bool_brown = false;

// color cloud

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> red_vector = std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> (2);
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> orange_vector = std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> (2);
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> yellow_vector = std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> (2);
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> green_vector = std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> (2);
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> blue_vector = std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> (2);
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> purple_vector = std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> (2);
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> brown_vector = std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> (2);
pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud (new pcl::PointCloud<pcl::PointXYZ>);

struct Rectangle
{   
    cv::Point u_l;
    cv::Point u_r;
    cv::Point l_l;
    cv::Point l_r;

    Rectangle(int x=0, int y=0)
    {   
        int x_val = x;
        int y_val = y;

        u_l.x = x_val, u_l.y = y_val;
        u_r.x = x_val + dim_w, u_r.y = y_val;
        l_l.x = x_val, l_l.y =  y_val + dim_h;
        l_r.x = x_val + dim_w, l_r.y = y_val + dim_h;
    }
};


class canvas{
public:
    bool setupQ;
    cv::Point origin;
    cv::Point corner;
    int minDims,maxDims;
    double scale;
    int rows, cols;
    cv::Mat img;
    void init(int minD, int maxD){
        // Initialise the canvas with minimum and maximum rows and column sizes.
        minDims = minD; maxDims = maxD;
        origin = cv::Point(0,0);
        corner = cv::Point(0,0);
        scale = 1.0;
        rows = 0;
        cols = 0;
        setupQ = false;
    }
    void stretch(cv::Point2f min, cv::Point2f max){
        // Stretch the canvas to include the points min and max.
        if(setupQ){
            if(corner.x < max.x){corner.x = (int)(max.x + 1.0);};
            if(corner.y < max.y){corner.y = (int)(max.y + 1.0);};
            if(origin.x > min.x){origin.x = (int) min.x;};
            if(origin.y > min.y){origin.y = (int) min.y;};
        } else {
            origin = cv::Point((int)min.x, (int)min.y);
            corner = cv::Point((int)(max.x + 1.0), (int)(max.y + 1.0));
        }
        int c = (int)(scale*((corner.x + 1.0) - origin.x));
        if(c<minDims){
            scale = scale * (double)minDims/(double)c;
        } else {
            if(c>maxDims){
                scale = scale * (double)maxDims/(double)c;
            }
        }
        int r = (int)(scale*((corner.y + 1.0) - origin.y));
        if(r<minDims){
            scale = scale * (double)minDims/(double)r;
        } else {
            if(r>maxDims){
                scale = scale * (double)maxDims/(double)r;
            }
        }
        cols = (int)(scale*((corner.x + 1.0) - origin.x));
        rows = (int)(scale*((corner.y + 1.0) - origin.y));
        setupQ = true;
    }
    void stretch(vector<Point2f> pts)
    {   // Stretch the canvas so all the points pts are on the canvas.
        cv::Point2f min = pts[0];
        cv::Point2f max = pts[0];
        for(size_t i=1; i < pts.size(); i++){
            Point2f pnt = pts[i];
            if(max.x < pnt.x){max.x = pnt.x;};
            if(max.y < pnt.y){max.y = pnt.y;};
            if(min.x > pnt.x){min.x = pnt.x;};
            if(min.y > pnt.y){min.y = pnt.y;};
        };
        stretch(min, max);
    }
    void stretch(cv::RotatedRect box)
    {   // Stretch the canvas so that the rectangle box is on the canvas.
        cv::Point2f min = box.center;
        cv::Point2f max = box.center;
        cv::Point2f vtx[4];
        box.points(vtx);
        for( int i = 0; i < 4; i++ ){
            cv::Point2f pnt = vtx[i];
            if(max.x < pnt.x){max.x = pnt.x;};
            if(max.y < pnt.y){max.y = pnt.y;};
            if(min.x > pnt.x){min.x = pnt.x;};
            if(min.y > pnt.y){min.y = pnt.y;};
        }
        stretch(min, max);
    }
    void drawEllipseWithBox(cv::RotatedRect box, cv::Scalar color, int lineThickness)
    {
        if(img.empty()){
            stretch(box);
            img = cv::Mat::zeros(rows,cols,CV_8UC3);
        }
        box.center = scale * cv::Point2f(box.center.x - origin.x, box.center.y - origin.y);
        box.size.width  = (float)(scale * box.size.width);
        box.size.height = (float)(scale * box.size.height);
        //ellipse(img, box, color, lineThickness, LINE_AA);
        Point2f vtx[4];
        box.points(vtx);
        for( int j = 0; j < 4; j++ ){
            line(img, vtx[j], vtx[(j+1)%4], color, lineThickness, LINE_AA);
        }
    }
    void drawPoints(vector<Point2f> pts, cv::Scalar color)
    {
        if(img.empty()){
            stretch(pts);
            img = cv::Mat::zeros(rows,cols,CV_8UC3);
        }
        for(size_t i=0; i < pts.size(); i++){
            Point2f pnt = scale * cv::Point2f(pts[i].x - origin.x, pts[i].y - origin.y);
            img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[0] = (uchar)color[0];
            img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[1] = (uchar)color[1];
            img.at<cv::Vec3b>(int(pnt.y), int(pnt.x))[2] = (uchar)color[2];
        };
    }
    void drawLabels( std::vector<std::string> text, std::vector<cv::Scalar> colors)
    {
        if(img.empty()){
            img = cv::Mat::zeros(rows,cols,CV_8UC3);
        }
        int vPos = 0;
        for (size_t i=0; i < text.size(); i++) {
            cv::Scalar color = colors[i];
            std::string txt = text[i];
            Size textsize = getTextSize(txt, FONT_HERSHEY_COMPLEX, 1, 1, 0);
            vPos += (int)(1.3 * textsize.height);
            Point org((img.cols - textsize.width), vPos);
            cv::putText(img, txt, org, FONT_HERSHEY_COMPLEX, 1, color, 1, LINE_8);
        }
    }
    void MergeImage( cv::Mat image )
    {
      for (int i = 0; i < image.rows; i++)
      {
        for (int j = 0; j < image.cols; j++)
        { 
          if (image.at<uchar>(i, j) == 0)
          {}
          else
          {
            img.at<cv::Vec3b>(i, j)[0] = 255;
            img.at<cv::Vec3b>(i, j)[1] = 255;
            img.at<cv::Vec3b>(i, j)[2] = 255;
          }
        }
      }
    }
};


namespace Plane {
    float DominantPlane::fx, DominantPlane::fy, DominantPlane::cx, DominantPlane::cy;
    float DominantPlane::DepthMapFactor;
    float DominantPlane::Distance_threshold;
    int DominantPlane::max_iter;
    int DominantPlane::N_pcds;
    int DominantPlane::frame_id;
    int DominantPlane::N_inlier;
    int DominantPlane::width;
    int DominantPlane::height;
    std::vector<cv::Vec3f> pcd_concat;
    cv::Mat DominantPlane::mK = cv::Mat::zeros(3, 3, CV_32FC1);
    cv::Mat DominantPlane::Pointcloud;

    DominantPlane::DominantPlane(float _fx, float _fy, float _cx, float _cy, float _DepthMapFactor, float _Distance_thresh,  int _max_iter,
                                 int _width, int _height) {
        mK.at<float>(0,0) = _fx;
        mK.at<float>(1,1) = _fy;
        mK.at<float>(0,2) = _cx;
        mK.at<float>(1,2) = _cy;
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
        DepthMapFactor = _DepthMapFactor;
        max_iter = _max_iter;
        frame_id = 0;
        N_inlier = 0;
        width = _width;
        height = _height;
        Distance_threshold = _Distance_thresh;
        Pointcloud = cv::Mat::zeros(height, width, CV_32FC3);
    }

    cv::Mat DominantPlane::Depth2pcd(cv::Mat &depth){
        float X, Y, Z;
        //cout << depth.size << endl;
        for (int y = 0; y < depth.rows; y++)
        {
            for(int x = 0; x < depth.cols; x++)
            {
                Z = depth.at<uint16_t>(y,x) / DepthMapFactor;
                if(Z > 0)
                {
                    X = (x - cx) * Z / fx;
                    Y = (y - cy) * Z / fy;
                    Pointcloud.at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
                    pcd_concat.push_back(cv::Vec3f(X,Y,Z));
                }
                else{
                     Pointcloud.at<cv::Vec3f>(y, x) = cv::Vec3f(0.f, 0.f, 0.f);
                }
            }
        }

        return Pointcloud;
    }
    

    
    Plane_model DominantPlane::RunRansac(cv::Mat &pcd_inlier, cv::Mat point_cloud) {
        unsigned long rand_idx;
        float cost = 0;
        float best_cost = std::numeric_limits<float>::infinity();
        unsigned int best_N_inlier = 0;
        Plane_model best_plane;
        std::srand(time(NULL));

        // cout << "frame id:" << frame_id << " Number of pcds:" <<  pcd_concat.size() << endl;
        Plane_model plane;

        for (int i = 0; i < max_iter; i++)
        {
            cv::Mat pcd_temp_inlier = cv::Mat::zeros(height, width, CV_32FC3);
            std::vector<cv::Vec3f> sampled_pcd;
            for(int j =0; j < 3; j++)
            {
                rand_idx = int(pcd_concat.size() * (std::rand()/(double)RAND_MAX));
                sampled_pcd.push_back(pcd_concat.at(rand_idx));
                // cout << "sampled_pcd" << sampled_pcd[j] << endl;
            }
            FindPlaneFromSampledPcds(sampled_pcd, plane);
           // cout << plane.a << " " << plane.b << " "<< plane.c << " " << plane.denominator << " " << endl;
            compute_inlier_cost(plane, pcd_temp_inlier, point_cloud);
            cost = plane.avg_distance / N_inlier;
            if(best_cost > cost)
            {
                best_cost = cost;
                best_plane = plane;
                cout << "inlier1/" << "iter:" << i << " " << "cost:" << cost << " " << "average distance:" << plane.avg_distance << " " << endl;
                pcd_inlier = pcd_temp_inlier;
            }
            N_inlier = 0;
            pcd_temp_inlier.release();
            sampled_pcd.clear();
        }

        

        ResetValue();
        return best_plane;
    }
    

    void DominantPlane::FindPlaneFromSampledPcds(std::vector<cv::Vec3f> &sampled_pcds, Plane_model& plane)
    {
        cv::Vec3f A = sampled_pcds[0], B = sampled_pcds[1], C = sampled_pcds[2];
        // cout<< "A" << A << "B" << B << endl;
        cv::Vec3f AB = A - B;
        cv::Vec3f BC = B - C;
        cv::Vec3f normal = AB.cross(BC);

        plane.a = normal[0], plane.b = normal[1], plane.c = normal[2];
        plane.d = -plane.a * A[0] - plane.b * A[1] - plane.c * A[2];
        plane.denominator = sqrt(pow(plane.a, 2) + pow(plane.b, 2) + pow(plane.c, 2));
    }

    void DominantPlane::compute_inlier_cost(Plane_model& plane, cv::Mat& pcd_inlier, cv::Mat& pcd_input)
    {
        float dist_all = 0;
        for (int y = 0; y < pcd_input.rows; y++)
        {
            for(int x = 0; x < pcd_input.cols; x++)
            {
                cv::Vec3f temp = pcd_input.at<cv::Vec3f>(y,x);
                float dist = abs(plane.a * temp[0] + plane.b * temp[1] + plane.c * temp[2] + plane.d) / plane.denominator;
                // cout << temp[0] << " " << temp[1] << " " << temp[2] << endl;
                if(dist < Distance_threshold)
                {
                    pcd_inlier.at<cv::Vec3f>(y,x) = temp;
                    N_inlier++;
                    dist_all += dist;
                }
                else
                {
                    pcd_inlier.at<cv::Vec3f>(y,x) = cv::Vec3f(0.f, 0.f, 0.f);
                }

            }
        }
        plane.avg_distance = dist_all / N_inlier;
    }
}

cv::Mat DistantPlaneSegmentation(Plane::Plane_model plane, cv::Mat pCloud_outlier, float distance)
{
    /*
    object_cloud->width  = red_cloud->width;
    object_cloud->height = red_cloud->height;
    object_cloud->points.resize (object_cloud->width * object_cloud->height);
    */
    cv::Mat pcd_object = cv::Mat::zeros(pCloud_outlier.rows, pCloud_outlier.cols, CV_32FC3);

    float Threshold = 0.001;

    for (int i=0; i<pcd_object.rows; i++)
    {
        for(int j=0; j<pcd_object.cols; j++)
        {
            float x = pCloud_outlier.at<cv::Vec3f>(i, j)[0];
            float y = pCloud_outlier.at<cv::Vec3f>(i, j)[1];
            float z = pCloud_outlier.at<cv::Vec3f>(i, j)[2];

            float dist = abs(plane.a * x + plane.b * y + plane.c * z + plane.d) / plane.denominator;

            if ( -Threshold + distance < dist && dist < Threshold + distance)
            {
                pcd_object.at<cv::Vec3f>(i, j) = pCloud_outlier.at<cv::Vec3f>(i, j);

            }
        }

    }
    return pcd_object;
    
}

void Segmentation(cv::Mat& image_RGB, cv::Mat& image_depth, Plane::Plane_model &best_plane, cv::Mat &pCloud_outlier)
{
    float fx = 615.6707153320312;
    float fy = 615.962158203125;
    float cx = 328.0010681152344;
    float cy = 241.31031799316406;

    float scale = 1000;
    float Distance_theshold = 0.0005;

    int width = 640;
    int height = 480;
    int max_iter = 300;

    Plane::DominantPlane plane(fx, fy, cx, cy, scale, Distance_theshold, max_iter, width, height);
    
    cv::Mat pCloud_inlier(height, width, CV_32FC3); 
    cv::Mat pCloud(height, width, CV_32FC3);
    
    pCloud = plane.Depth2pcd(image_depth);

    Plane::Plane_model dominant_plane;
    dominant_plane = plane.RunRansac(pCloud_inlier, pCloud);

    best_plane = dominant_plane;

    for (int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            pCloud_outlier.at<cv::Vec3f>(y, x) = pCloud.at<cv::Vec3f>(y, x) - pCloud_inlier.at<cv::Vec3f>(y, x);
        }
    }
}

void Color_Segmentation(cv::Mat& RGB_image, cv::Mat& pCloud)
{   
    Mat im = RGB_image.clone();
    Mat hsv_image;
    cvtColor(im, hsv_image, COLOR_BGR2HSV); // convert BGR2HSV

    Mat lower_red_hue;
    Mat upper_red_hue;

    inRange(hsv_image, Scalar(0, 100, 50), Scalar(3, 255, 255), lower_red_hue);
    inRange(hsv_image, Scalar(150, 100, 50), Scalar(179, 255, 255), upper_red_hue);

    Mat red;
    addWeighted(lower_red_hue, 1.0, upper_red_hue, 1.0, 0.0, red);

        // Threshold for orange color
    Mat orange;
    inRange(hsv_image, Scalar(4, 100, 30), Scalar(10, 255,200), orange);

        // Threshold for yellow color
    Mat yellow;
    inRange(hsv_image, Scalar(11, 200, 50), Scalar(25, 255, 255), yellow);

        // Threshold for green color
    Mat green;
    inRange(hsv_image, Scalar(50, 40, 40), Scalar(90, 255, 255), green);

        // Threshold for blue color
    Mat blue;
    inRange(hsv_image, Scalar(102, 100, 40), Scalar(130, 255, 255), blue);

        // Threshold for purple color. the hue for purple is the same as red. Only difference is value.
    Mat purple;
    inRange(hsv_image, Scalar(131, 50, 30), Scalar(179, 255, 140), purple);

        // Threshold for brown color. the hue for brown is the same as red and orange. Only difference is value.
    Mat brown;
    inRange(hsv_image, Scalar(0, 50, 10), Scalar(15, 200, 100), brown);

    bool_red = colorCheck(red, red_pCloud, pCloud);
    bool_orange = colorCheck(orange, orange_pCloud, pCloud);
    bool_yellow = colorCheck(yellow, yellow_pCloud, pCloud);
    bool_green = colorCheck(green, green_pCloud, pCloud);
    bool_blue = colorCheck(blue, blue_pCloud, pCloud);
    bool_purple = colorCheck(purple, purple_pCloud, pCloud);
    bool_brown = colorCheck(brown, brown_pCloud, pCloud);
}

bool colorCheck(cv::Mat color_image, cv::Mat& color_pCloud, cv::Mat pCloud)
{   
    bool bool_color = false;
    color_pCloud = pCloud.clone();

    int color_num = 0;
    for (int i = 0; i < color_image.rows; i ++)
    {
        for (int j = 0; j < color_image.cols; j++)
        {
            if(color_image.at<uchar>(i, j) == 255)
                { color_num++; }
        }
    }
    if(!color_num == 0)
    {   
        bool_color = true;
        for (int i = 0; i < color_image.rows; i++)
        {
            for (int j = 0; j < color_image.cols; j++)
            {
                if (color_image.at<uchar>(i, j) == 0) // ******choose color****** //
                {
                    for (int k = 0; k < 3 ; k++)
                    color_pCloud.at<cv::Vec3f>(i, j)[k] = 0;
                } 
            }
        }
    }

    return bool_color;
}

void LoadImages(const string &association_dir, vector<string> &FilenameRGB, vector<string> &FilenameDepth)
{
    ifstream association;
    association.open(association_dir.c_str());
    while(!association.eof())
    {
        string s;
        getline(association, s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string sRGB, sDepth;
            ss >> sRGB;
            FilenameRGB.push_back(sRGB);
            ss >> sDepth;
            FilenameDepth.push_back(sDepth);
        }
    }
}


int main(int argc, char** argv)
{      
    if(argc != 3)
    {
        cout << "Usage: ./matching_algorithm <DatasetDir> <association file dir>" << endl;
    }

    string image_dir = argv[1];
    string association_dir = argv[2];
    vector<string> FilenameRGB;
    vector<string> FilenameDepth;
    LoadImages(association_dir, FilenameRGB, FilenameDepth);
    int nImages = FilenameRGB.size();

    if(FilenameRGB.empty())
    {
        cerr << endl << "No image, please give right path" << endl;
        return 1;
    }
    else if(FilenameRGB.size()!=FilenameDepth.size())
    {
        cerr << endl << "Different number of image for rgb and depth" << endl;
        return 1;
    }

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    for (int n = 0; n < nImages; n++)
    {   

        cv::Mat image_RGB;
        image_RGB = cv::imread(image_dir + "/" + FilenameRGB[n], IMREAD_UNCHANGED);

        cv::Mat image_depth;
        image_depth = cv::imread(image_dir + "/" + FilenameDepth[n], IMREAD_UNCHANGED);
        
        if(image_RGB.empty() || image_depth.empty())
        {
            cerr << endl << "Failed to load image at: " << FilenameRGB[n] << endl;
            return 1;
        }

        Plane::Plane_model dominant_plane;
        cv::Mat pCloud_outlier = cv::Mat::zeros(image_RGB.rows, image_RGB.cols, CV_32FC3);
        Segmentation(image_RGB, image_depth, dominant_plane, pCloud_outlier);

        Ineedoutlier = pCloud_outlier.clone();

        Color_Segmentation(image_RGB, pCloud_outlier);

        if(bool_red)
            { BoundingBox(dominant_plane, red_pCloud, "red"); }
        if(bool_orange)
            { BoundingBox(dominant_plane, orange_pCloud, "orange"); }
        if(bool_yellow)
            { BoundingBox(dominant_plane, yellow_pCloud, "yellow"); }
        if(bool_green)
            { BoundingBox(dominant_plane, green_pCloud, "green"); }
        if(bool_blue)
            { BoundingBox(dominant_plane, blue_pCloud, "blue"); }
        if(bool_purple)
            { BoundingBox(dominant_plane, purple_pCloud, "purple"); }
        if(bool_brown)
            { BoundingBox(dominant_plane, brown_pCloud, "brown"); }
        

        if(bool_red)
            { pcl::PointCloud<pcl::PointXYZ> red_synthetic = makeSyntheticCloud("red"); }
        if(bool_orange)
            { pcl::PointCloud<pcl::PointXYZ> orange_synthetic = makeSyntheticCloud("orange"); }
        if(bool_yellow)
            { pcl::PointCloud<pcl::PointXYZ> yellow_synthetic = makeSyntheticCloud("yellow"); }
        if(bool_green)
            { pcl::PointCloud<pcl::PointXYZ> green_synthetic = makeSyntheticCloud("green"); }
        if(bool_blue)
            { pcl::PointCloud<pcl::PointXYZ> blue_synthetic = makeSyntheticCloud("blue"); }
        if(bool_purple)
            { pcl::PointCloud<pcl::PointXYZ> purple_synthetic = makeSyntheticCloud("purple"); }
        if(bool_brown)
            { pcl::PointCloud<pcl::PointXYZ> brown_synthetic = makeSyntheticCloud("brown"); }

        /*
        if(bool_red)
            { Matrix red_RT = GetPose("red", red_synthetic); }
        if(bool_orange)
            { Matrix orange_RT = GetPose("orange", orange_synthetic); }
        if(bool_yellow)
            { Matrix yellow_RT = GetPose("yellow", yellow_synthetic); }
        if(bool_green)
            { Matrix green_RT = GetPose("green", green_synthetic); }
        if(bool_blue)
            { Matrix blue_RT = GetPose("blue", blue_synthetic); }
        if(bool_purple)
            { Matrix purple_RT = GetPose("purple", purple_synthetic); }
        if(bool_brown)
            { Matrix brown_RT = GetPose("brown", brown_synthetic); }
        */

        if(bool_red)
            { Merge(red_vector); }
        if(bool_orange)
            { Merge(orange_vector); }
        if(bool_yellow)
            { Merge(yellow_vector); }
        if(bool_green)
            { Merge(green_vector); }
        if(bool_blue)
            { Merge(blue_vector); }
        if(bool_purple)
            { Merge(purple_vector); }
        if(bool_brown)
            { Merge(brown_vector); }

        //string name = "merged_cloud_" + to_string(n) + ".ply";
        //pcl::io::savePLYFileASCII (name, *merged_cloud);

        vizPointCloud(image_RGB, merged_cloud);
        merged_cloud->points.clear();
    }
    
    /*
    cv::Mat image_RGB;
    image_RGB = cv::imread("image_RGB.jpg", CV_LOAD_IMAGE_ANYCOLOR);

    cv::Mat image_depth;
    image_depth = cv::imread("image_depth.png", CV_LOAD_IMAGE_ANYDEPTH);
        
    Plane::Plane_model dominant_plane;
    cv::Mat pCloud_outlier = cv::Mat::zeros(image_RGB.rows, image_RGB.cols, CV_32FC3);
    Segmentation(image_RGB, image_depth, dominant_plane, pCloud_outlier);

    Color_Segmentation(image_RGB, pCloud_outlier);

    if(bool_red)
        { BoundingBox(dominant_plane, red_pCloud, "red"); }
    if(bool_orange)
        { BoundingBox(dominant_plane, orange_pCloud, "orange"); }
    if(bool_yellow)
        { BoundingBox(dominant_plane, yellow_pCloud, "yellow"); }
    if(bool_green)
        { BoundingBox(dominant_plane, green_pCloud, "green"); }
    if(bool_blue)
        { BoundingBox(dominant_plane, blue_pCloud, "blue"); }
    if(bool_purple)
        { BoundingBox(dominant_plane, purple_pCloud, "purple"); }
    if(bool_brown)
        { BoundingBox(dominant_plane, brown_pCloud, "brown"); }
    

    if(bool_red)
        { pcl::PointCloud<pcl::PointXYZ> red_synthetic = makeSyntheticCloud("red"); }
    if(bool_orange)
        { pcl::PointCloud<pcl::PointXYZ> orange_synthetic = makeSyntheticCloud("orange"); }
    if(bool_yellow)
        { pcl::PointCloud<pcl::PointXYZ> yellow_synthetic = makeSyntheticCloud("yellow"); }
    if(bool_green)
        { pcl::PointCloud<pcl::PointXYZ> green_synthetic = makeSyntheticCloud("green"); }
    if(bool_blue)
        { pcl::PointCloud<pcl::PointXYZ> blue_synthetic = makeSyntheticCloud("blue"); }
    if(bool_purple)
        { pcl::PointCloud<pcl::PointXYZ> purple_synthetic = makeSyntheticCloud("purple"); }
    if(bool_brown)
        { pcl::PointCloud<pcl::PointXYZ> brown_synthetic = makeSyntheticCloud("brown"); }

    if(bool_red)
        { Merge(red_vector); }
    if(bool_orange)
        { Merge(orange_vector); }
    if(bool_yellow)
        { Merge(yellow_vector); }
    if(bool_green)
        { Merge(green_vector); }
    if(bool_blue)
        { Merge(blue_vector); }
    if(bool_purple)
        { Merge(purple_vector); }
    if(bool_brown)
        { Merge(brown_vector); }

    pcl::io::savePLYFileASCII ("merged_cloud.ply", *merged_cloud);
    
    vizPointCloud(image_RGB, merged_cloud);
    merged_cloud->points.clear();
    */
    return (0);
}



void BoundingBox(Plane::Plane_model dominant_plane, cv::Mat pCloud_outlier, std::string color_string)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    outlier_cloud = cv2pcl(pCloud_outlier);

    block_height = GetDist(dominant_plane, outlier_cloud); // SSG
    
    if(color_string == "red")
        { red_block_height = block_height;}
    else if(color_string == "orange")
        { orange_block_height = block_height;}
    else if(color_string == "yellow")
        { yellow_block_height = block_height;}
    else if(color_string == "green")
        { green_block_height = block_height;}
    else if(color_string == "blue")
        { blue_block_height = block_height;}
    else if(color_string == "purple")
        { purple_block_height = block_height;}
    else
        { brown_block_height = block_height;}

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_projected_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl_projected_cloud = pcl_dominant_plane_projection(dominant_plane, pCloud_outlier);

    cv::Mat projected_image = get_projected_image(dominant_plane, pcl_projected_cloud);

    cout << " What is the problem? two" << endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr bbox_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    cout << " What is the problem? one " << endl;

    if(!projected_image.empty())
    {
        cout << "projected image: " << projected_image.rows << ",  " << projected_image.cols << endl;
        cv::Mat box_image = processImage(projected_image, color_string);
        box_cloud = back_projection(box_image, color_string);
        bbox_cloud = makeForm(box_cloud);

    }

    cout << " What is the problem? " << endl;

    //CloudViewer(outlier_cloud, bbox_cloud);

    if(color_string == "red")
    {
        red_vector.at(0) = outlier_cloud;
        red_vector.at(1) = bbox_cloud;
    }
    else if(color_string == "orange")
    {
        orange_vector.at(0) = outlier_cloud;
        orange_vector.at(1) = bbox_cloud;        
    }
    else if(color_string == "yellow")
    {
        yellow_vector.at(0) = outlier_cloud;
        yellow_vector.at(1) = bbox_cloud;        
    }
    else if(color_string == "green")
    {
        green_vector.at(0) = outlier_cloud;
        green_vector.at(1) = bbox_cloud;        
    }
    else if(color_string == "blue")
    {
        blue_vector.at(0) = outlier_cloud;
        blue_vector.at(1) = bbox_cloud;        
    }
    else if(color_string == "purple")
    {
        purple_vector.at(0) = outlier_cloud;
        purple_vector.at(1) = bbox_cloud;        
    }
    else
    {
        brown_vector.at(0) = outlier_cloud;
        brown_vector.at(1) = bbox_cloud;        
    }
}


pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_dominant_plane_projection(Plane::Plane_model plane, cv::Mat pCloud)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_projected_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    float a = plane.a; 
    float b = plane.b; 
    float c = plane.c; 
    float d = plane.d;

    for(int i = 0; i < pCloud.rows; i++)
    {
        for(int j = 0; j < pCloud.cols; j++)
        {
            if (pCloud.at<cv::Vec3f>(i, j)[0] == 0 && pCloud.at<cv::Vec3f>(i, j)[1] == 0 && pCloud.at<cv::Vec3f>(i, j)[2] == 0)
            {}
            else   
            {
                double x = pCloud.at<cv::Vec3f>(i, j)[0];
                double y = pCloud.at<cv::Vec3f>(i, j)[1];
                double z = pCloud.at<cv::Vec3f>(i, j)[2];
        
                double t = (- a*x - b*y - c*z - d)/(a*a + b*b + c*c);

                pcl::PointXYZ point;
                point.x = x + t*a;
                point.y = y + t*b;
                point.z = z + t*c;

                pcl_projected_cloud->points.push_back(point);
            }
        }
    }

    return pcl_projected_cloud;
}


cv::Mat get_projected_image(Plane::Plane_model plane, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
{       
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (point_cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);

    //std::cerr << "Cloud after filtering: " << std::endl;
    //std::cerr << *cloud_filtered << std::endl;
    
    double x_min = 100;
    double y_x;
    double z_x;

    double y_min = 100;
    double x_y;
    double z_y;

    double z_min = 100;
    double x_z;
    double y_z;

    int num = 0;

    for (int i = 0; i < cloud_filtered->points.size(); i++)
    {
        pcl::PointXYZ point = cloud_filtered->points[i];

        double x = point.x;
        double y = point.y;
        double z = point.z;

        if ( x < x_min )
        {
            x_min = x;
            y_x = y;
            z_x = z;
        }
        if ( y < y_min )
        {
            y_min = y;
            x_y = x;
            z_y = z;
        }
        if ( z < z_min )
        {
            z_min = z;
            x_z = x;
            y_z = y;
        }

        
    }

    cout << " I'm the error " << endl;

    origin.x = x_min;
    origin.y = y_min; // hyperparameter
    origin.z = (- x_min*plane.a - y_min*plane.b - plane.d) / plane.c; 

    pcl::PointXYZ x_dir;
    x_dir.x = x_min - origin.x;
    x_dir.y = y_x - origin.y;
    x_dir.z = z_x - origin.z;

    double norm = sqrt(pow(x_dir.x, 2) + pow(x_dir.y, 2) + pow(x_dir.z, 2)) + 1e-10;

    // How about adding '1e-10'?

    unit_x.x = x_dir.x / norm;
    unit_x.y = x_dir.y / norm;
    unit_x.z = x_dir.z / norm;

    a = plane.a;
    b = plane.b;
    c = plane.c;
    d = plane.d;  

    if(c > 0)
    {
        a = - a; 
        b = - b; 
        c = - c; 
        d = - d; 
    }

    unit_y = GetUnitY(origin, unit_x, plane);

    cout << " unit_y value : " << unit_y.x << ", " << unit_y.y << ", " << unit_y.z << endl;
    /*    
    unit_y.x = 0.9948;
    unit_y.y = -0.0786;
    unit_y.z = -0.0652;
    */
    
    double max_x = 0;
    double max_y = 0;
    double min_x = 10;
    double min_y = 10;

    for (int i = 0; i < cloud_filtered->points.size(); i++)
    {
        pcl::PointXYZ point = cloud_filtered->points[i];
        point.x -= origin.x;
        point.y -= origin.y;
        point.z -= origin.z;

        int x = (unit_x.x*point.x + unit_x.y*point.y + unit_x.z*point.z)*1000;
        int y = (unit_y.x*point.x + unit_y.y*point.y + unit_y.z*point.z)*1000;

        if(max_x < x) {max_x = x;};
        if(max_y < y) {max_y = y;};
        if(min_x > x) {min_x = x;};
        if(min_y > y) {min_y = y;};
    }

    if(((max_x - min_x) > 0) && ((max_y - min_y) > 0) && ((max_y - min_y) < 1e+5))
    { 
        cout << max_x - min_x << ", " << max_y - min_y << endl;

        cv::Mat projected_image = cv::Mat::zeros(max_x - min_x + 10, max_y - min_y + 10, CV_8UC1); 

        for (int i = 0; i < cloud_filtered->points.size(); i++)
        {
            pcl::PointXYZ point = cloud_filtered->points[i];
            point.x -= origin.x;
            point.y -= origin.y;
            point.z -= origin.z;

            int x = (unit_x.x*point.x + unit_x.y*point.y + unit_x.z*point.z)*1000 - min_x + 5;
            int y = (unit_y.x*point.x + unit_y.y*point.y + unit_y.z*point.z)*1000 - min_y + 5;
            
            /*
            if(y < 0)
            {
                y = - y;
                //minus_y = true;
            }

            if(x < 0)
            {
                x = -x;
                //minus_x = true;
            }
            */
            cout << " where is the problem? " << endl;

            if((max_x - min_x + 10 > x) && (x > 0) && (max_y - min_y + 10 > y) && (y > 0))
                { projected_image.at<uchar>(x, y) = 255; }
        }

        cout << " where is the problem?! " << endl;

        return projected_image;
    }

    /*
    float fx = 612.86083984375;
    float fy = 613.0430908203125;
    float cx = 316.27764892578125; 
    float cy = 250.1717071533203;
    int width = 640;
    int height = 480;

    cv::Mat projected_image = cv::Mat::zeros(width, height, CV_8UC3);

    for (int i =0; i < point_cloud->size(); i++)
    {
        pcl::PointXYZ temp_cloud; 
        temp_cloud = point_cloud->points[i];
        float X = temp_cloud.x;
        float Y = temp_cloud.y; 
        float Z = temp_cloud.z; 
        int x = cy + fy * Y / Z;
        int y = cx + fx * X / Z;
        projected_image.at<Vec3b>(x, y)[2] = 255;
    }
    */
    
}   



void CloudViewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2)
{
    // viewer

    pcl::visualization::PCLVisualizer viewer ("Cloud Viewer");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_1_color_handler (cloud_1, 255, 255, 255);
    viewer.addPointCloud (cloud_1, cloud_1_color_handler, "cloud_1");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_2_color_handler (cloud_2, 230, 20, 20); // Red
    viewer.addPointCloud (cloud_2, cloud_2_color_handler, "cloud_2");

    //viewer.addCoordinateSystem (1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_1");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_2");
    //viewer.setPosition(800, 400); // Setting visualiser window position

    //viewer.spinOnce(1);
    
    while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
      viewer.spinOnce ();
    }
    
}

cv::Mat processImage(cv::Mat image, std::string string)
{       
    int thresh = 100;
    RNG rng(12345);

    blur( image, image, Size(3,3) );
    /*
    Mat canny_output;
    Canny( image, canny_output, thresh, thresh*2, 3 );
    vector<vector<Point> > contours;
    */
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using Threshold
    threshold( image, threshold_output, thresh, 255, THRESH_BINARY );

    findContours( threshold_output, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    int max_num = 0;
    int max_index = 0;

    cout << " Maybe here? " << endl;

    

    cout << " Maybe... " << endl;

    Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC1);

    cout << " Maybe...?! " << endl;

    if(!contours.empty())
    {   
        for(int i = 0; i < contours.size(); i++)
        {
            if(contours[i].size() > max_num)
               { max_num = contours[i].size(); max_index = i;}
        }

        cout << "What?: " << !contours.empty() << endl;

        RotatedRect minRect = minAreaRect( Mat(contours[max_index]) );

        cout << "What happened?" << endl;

        Point2f rect_points[4];
        minRect.points( rect_points );

        for ( int j = 0; j < 4; j++ )
        {   
            rectpoints.at(j) = rect_points[j];
        }
        
        //RNG rng(12345);

        //Point vertices[4];
        for ( int j = 0; j < 4; j++ )
        {   
            line( drawing, rectpoints.at(j), rectpoints.at((j+1)%4), 255 );
        } 
        //cv::Mat output_image = RectFitting(image, string);    
    }
    
    return drawing;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr back_projection(cv::Mat image, std::string string)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if(image.at<uchar>(i, j) == 0)
            {}
            else 
            {   
                
                double i_ = i + min_x - 5;
                double j_ = j + min_y - 5;

                i_ /= 1000;
                j_ /= 1000;
                

                /*
                if (minus_x == true)
                {
                    i_ = - i_;
                }

                if (minus_y == true)
                {
                    j_ = - j_;
                }
                
                /*
                Z_deominator = a * (j - cx) / fx + b * (i - cy) / fy + c;
                double z = - d / Z_deominator;
                double x = (j - cx) / fx * z;//i_*(unit_x.x) + j_*(unit_y.x) + origin.x;
                double y = (i - cy) / fy * z; //i_*(unit_x.y) + j_*(unit_y.y) + origin.y;
                //double z = //i_*(unit_x.z) + j_*(unit_y.z) + origin.z;
                */

                double x = i_*(unit_x.x) + j_*(unit_y.x) + origin.x;
                double y = i_*(unit_x.y) + j_*(unit_y.y) + origin.y;
                double z = i_*(unit_x.z) + j_*(unit_y.z) + origin.z;

                pcl::PointXYZ point;
                point.x = x;
                point.y = y;
                point.z = z;

                box_cloud->points.push_back(point);
            }
        }
    }

    for (int i = 0; i < 4; i++)
    {
        double i_ = rectpoints.at(i).y + min_x - 5;
        double j_ = rectpoints.at(i).x + min_y - 5;

        i_ /= 1000;
        j_ /= 1000;
                

        //Z_deominator = a * (j_ - cx) / fx + b * (i_ - cy) / fy + c;
        //double z = //- d / Z_deominator;
        //double x = //(j_ - cx) / fx * z;
        //double y = //(i_ - cy) / fy * z;


        double x = i_*(unit_x.x) + j_*(unit_y.x) + origin.x;
        double y = i_*(unit_x.y) + j_*(unit_y.y) + origin.y;
        double z = i_*(unit_x.z) + j_*(unit_y.z) + origin.z;

        keypoints.at(i).x = x;
        keypoints.at(i).y = y;
        keypoints.at(i).z = z;

        if(string == "red")
        {
            red_keypoints.at(i).x = x;
            red_keypoints.at(i).y = y;
            red_keypoints.at(i).z = z;
    
        }
        else if (string == "orange")
        {
            orange_keypoints.at(i).x = x;
            orange_keypoints.at(i).y = y;
            orange_keypoints.at(i).z = z;        
        }
        else if (string == "yellow")
        {
            yellow_keypoints.at(i).x = x;
            yellow_keypoints.at(i).y = y;
            yellow_keypoints.at(i).z = z;        
        }
        else if (string == "green")
        {
            green_keypoints.at(i).x = x;
            green_keypoints.at(i).y = y;
            green_keypoints.at(i).z = z;        
        }
        else if (string == "blue")
        {
            blue_keypoints.at(i).x = x;
            blue_keypoints.at(i).y = y;
            blue_keypoints.at(i).z = z;        
        }
        else if (string == "purple")
        {
            purple_keypoints.at(i).x = x;
            purple_keypoints.at(i).y = y;
            purple_keypoints.at(i).z = z;        
        }
        else
        {
            brown_keypoints.at(i).x = x;
            brown_keypoints.at(i).y = y;
            brown_keypoints.at(i).z = z;        
        }
    }

    return box_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr makeForm (pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp (new pcl::PointCloud<pcl::PointXYZ>);
    float norm = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));

    unit_vector.at(0) = a / norm;
    unit_vector.at(1) = b / norm;
    unit_vector.at(2) = c / norm;

    for (int i = 0; i < box_cloud->points.size(); i++)
    {
        pcl::PointXYZ point = box_cloud->points[i];
        double x = point.x;
        double y = point.y;
        double z = point.z;

        //GetDist()
        double scalar = block_height;

        pcl::PointXYZ pt;
        pt.x = x + scalar*unit_vector.at(0);
        pt.y = y + scalar*unit_vector.at(1);
        pt.z = z + scalar*unit_vector.at(2);

        temp->points.push_back(pt);
    }

    int num = 0;

    for(int i = 0; i < box_cloud->points.size(); i++)
    {
        pcl::PointXYZ point = box_cloud->points[i];
        temp->points.push_back(point);
    }

    double val = 1e-8; // hyper parameter

    for (int i = 0; i < 4; i++)
    {   
        pcl::PointXYZ pts;
        pts.x = keypoints.at(i).x;
        pts.y = keypoints.at(i).y;
        pts.z = keypoints.at(i).z;

        //GetDist()
        double intval = block_height / 20;

        for (int j = 0; j < 20; j++)
        {   
            pcl::PointXYZ pt;
            pt.x = intval*j*unit_vector.at(0) + pts.x;
            pt.y = intval*j*unit_vector.at(1) + pts.y;
            pt.z = intval*j*unit_vector.at(2) + pts.z;

            temp->points.push_back(pt);
        }
    }    

    return temp;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr cv2pcl(cv::Mat image)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {   
            pcl::PointXYZ point;
            point.x = image.at<cv::Vec3f>(i, j)[0];
            point.y = image.at<cv::Vec3f>(i, j)[1];
            point.z = image.at<cv::Vec3f>(i, j)[2];

            cloud->points.push_back(point);
        }
    }

    return cloud;
}

pcl::PointXYZ GetUnitY(pcl::PointXYZ origin, pcl::PointXYZ unit_x, Plane::Plane_model plane)
{   
    pcl::PointXYZ unit_y;

    if( !b == 0 )
    {
        float K = - c - origin.x * a - origin.y * b - origin.z * c - d;

        unit_y.x = (- unit_x.z - unit_x.y * K / b) / (unit_x.x + unit_x.y * (- a / b));
        unit_y.y = (K - a * unit_y.x) / b;
        unit_y.z = 1;
   
    }else if ( !a == 0 )
    {
        float K = - c - origin.x * a - origin.z * c - d;
        unit_y.x = K / a;
        unit_y.y = ( - unit_x.z - unit_x.x * unit_y.x ) / unit_x.y;
        unit_y.z = 1;
    }else
    {   
        unit_y.x = 1;
        unit_y.z = (- d / c) - origin.z;
        unit_y.y = (- unit_x.z * unit_y.z - unit_x.x) / unit_x.y;
    }

    
    float norm = sqrt(pow(unit_y.x, 2) + pow(unit_y.y, 2) + pow(unit_y.z, 2));

    unit_y.x /= norm;
    unit_y.y /= norm;
    unit_y.z /= norm;

    if(unit_y.x < 0)
    {
        unit_y.x = - unit_y.x;
        unit_y.y = - unit_y.y;
        unit_y.z = - unit_y.z;
    }

    return unit_y;
}

cv::Mat RectFitting(cv::Mat projected_image, std::string string)
{   
    cv::Mat image; 
    image = projected_image.clone();
    //cv::resize(projected_image, image, cv::Size(projected_image.cols*2, projected_image.rows*2), 0, 0, cv::INTER_CUBIC);  

    //cv::Mat contour_image;
    //cv::resize(lineFitting(projected_image), contour_image, cv::Size(projected_image.cols*2, projected_image.rows*2), 0, 0, CV_INTER_LANCZOS4);  

    int origin_num = 0;
    
    int min_x = 0;
    int min_y = 0;

    int max_x = 0;
    int max_y = 0;

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            if(image.at<uchar>(y, x) == 255)
            {
                if(x < min_x)
                    { min_x = x; };

                if(y < min_y)
                    { min_y = y; };

                if(x > max_x)
                    { max_x = x; };

                if(y > max_y)
                    { max_y = y; };

                origin_num++;
            }
        }
    } 

    int dim_w, dim_h;
    
    if(string == "red")
    {   
        dim_w = 50; // We have to change the value
        dim_h = 25; // We have to change the value
    }
    else if(string == "yellow")
    {
        dim_w = 75;
        dim_h = 25;
    }
    else if(string == "orange")
    {
        dim_w = 75;
        dim_h = 25;
    }
    else if(string == "green")
    {
        dim_w = 75;
        dim_h = 25;
    }
    else if(string == "blue")
    {
        dim_w = 50;
        dim_h = 50;
    }
    else if(string == "purple")
    {
        dim_w = 50;
        dim_h = 50;
    }
    else
    {
        dim_w = 50;
        dim_h = 50;
    }
    
    int num = 0;
    int best_num = 0;
    int x_sum = 0;
    int y_sum = 0;

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            if(image.at<uchar>(y, x) == 255)
            {   
                num += 1;
                x_sum += x;
                y_sum += y;
            }
        }
    } 
    
    cv::Point rect_cent((int)x_sum / num, (int)y_sum / num); //(max_x + min_x) / 2, (int)(max_y + min_y) / 2);
    cv::Point2f cent(rect_cent.x, rect_cent.y);
    Rectangle rect(rect_cent.x - dim_w/2, rect_cent.y - dim_h/2);

    int min_area;
    int max_area;

     if(string == "red")
    {   
        min_area = 1300;
        max_area = 1100;
    }
     else if(string == "yellow" || string == "orange" || string == "green")
    {   
        min_area = 1900;
        max_area = 1700;
    }
    else
    {
        min_area = 2600;
        max_area = 2400;
    }


    int best_theta;
    Rect best_b_box;

    for (int theta = 0; theta < 36000; theta++)
    {   
        num = 0;

        Mat rotation = getRotationMatrix2D(cent, theta*0.01, 1);
        Mat rotated_image;
        warpAffine(image, rotated_image, rotation, cv::Size(image.cols, image.rows));
 
        Rect b_box = lineFitting(rotated_image);
        
        for (int i = b_box.x; i < b_box.x + b_box.width; i++)
        {
            for (int j = b_box.y; j < b_box.y + b_box.height; j++)
            {
                if(rotated_image.at<uchar>(j, i) == 255)
                    { num++; };
            }
        }
        
        /*
        for (int i = rect.u_l.y; i < rect.l_l.y; i++)
        {
            for (int j = rect.u_l.x; j < rect.u_r.x; j++)
            {
                if(rotated_image.at<uchar>(i, j) == 255)
                    { num++; };
            }
        }

        cv::Mat drawing_image = rotated_image.clone();
        rectangle(drawing_image, rect.u_l, rect.l_r, 255);

        namedWindow("drawing", WINDOW_AUTOSIZE);
        imshow("drawing", drawing_image);
        waitKey(2);
        */

        /*
        if(55 > b_box.width && b_box.width > 45 && 30 > b_box.height && b_box.height > 20)
        {   
            if(num > best_num)
            {
                cout << " I'm here~ "  << endl;
                cout << "number of inlier: " << num << endl;

                best_num = num;
                best_theta = theta*0.01;

                best_b_box = b_box;
        
            }
        }
        */
        if(num > best_num && b_box.area() < min_area && b_box.area() > max_area)
        {
            min_area = b_box.area();
            best_theta = theta*0.01;
            best_b_box = b_box;
            best_num = num;
        }
    }

    cout << "best_theta : " << best_theta << endl;
    cout << "minimum_area : " << min_area << endl;
    //cout << "origin_num : " << origin_num << endl;

    //int rect_theta = 360 - best_theta;

    //bool minus_x = true;
    //bool minus_y = true;

    //dim_w /= 2;
    //dim_h /= 2;

    //cent.x /= 2;
    //cent.y /= 2;

    /*
    cv::Point2f vertices2f[4];
    cv::RotatedRect rotated_rect(cent, cv::Size(dim_w, dim_h), - best_theta);
    rotated_rect.points(vertices2f);  

 
    for (int i = 0; i < 4; i ++)
    {   
        vertices[i] = vertices2f[i];
        cout << vertices[i] << endl;    
    }
    */
    
    
    Mat drawing = Mat::zeros( image.size(), CV_8UC1 );
    rectangle( drawing, best_b_box.tl(), best_b_box.br(), 255, 1, 8, 0 );

    Mat rotation = getRotationMatrix2D(cent, - best_theta, 1);
    Mat rotated_b_box;
    warpAffine(drawing, rotated_b_box, rotation, cv::Size(drawing.cols, drawing.rows));

    rect_points[0] = Rotate(best_b_box.x, best_b_box.y, cent.x, cent.y, - best_theta);
    rect_points[1] = Rotate(best_b_box.x + best_b_box.width, best_b_box.y, cent.x, cent.y, - best_theta);
    rect_points[2] = Rotate(best_b_box.x + best_b_box.width, best_b_box.y + best_b_box.height, cent.x, cent.y, - best_theta);
    rect_points[3] = Rotate(best_b_box.x, best_b_box.y + best_b_box.height, cent.x, cent.y, - best_theta);

    return rotated_b_box;
    

    /*
    while(minus_x || minus_y)
    {   
        cv::RotatedRect rotated_rect(cent, cv::Size(dim_h, dim_w), best_theta);

        rotated_rect.points(vertices2f);    

        if(abs(vertices2f[0].x) > 1000 || abs(vertices2f[0].y) > 10000)    
                { break; };
        if(abs(vertices2f[1].x) > 1000 || abs(vertices2f[1].y) > 10000)    
                { break; };
        if(abs(vertices2f[2].x) > 1000 || abs(vertices2f[2].y) > 10000)    
                { break; };
        if(abs(vertices2f[3].x) > 10000 || abs(vertices2f[3].y) > 10000)    
                { break; };

        for (int i = 0; i < 4; i ++)
        {   
            vertices[i] = vertices2f[i];
            cout << vertices[i] << endl;    
        }

        cout << "\n" << endl;

        int min_x = 0;
        int min_y = 0;

        for (int i = 0; i < 4; i ++)
        {
            if(vertices[i].x < min_x)
                { min_x = vertices[i].x; };
            if(vertices[i].y < min_y)
                { min_y = vertices[i].y; };
        }   

        minus_x = min_x < 0;
        minus_y = min_y < 0;

        if(minus_x == true)
        {
            dim_w -= 1;
        }

        if(minus_y == true)
        {
            dim_h -= 1;
        }

    }
    */
        
    /*
    dim_w = 50; // We have to change the value
    dim_h = 25; // We have to change the value

    int h = projected_image.rows;
    int w = projected_image.cols;

    int best_num = 0;
    int best_theta;

    Point best_cent;

    for (int theta = 0; theta < 180; theta++)
    {   
        for (int i = 0; i < h - dim_h/2; i++)
        {
            for (int j = 0; j < w - dim_w/2; j++)
            {   
                int num = 0;


                Point cent(j + dim_w/2, i + dim_h/2);
                Mat rotation = getRotationMatrix2D(cent, theta, 2);
                Mat rotated_image;
                warpAffine(projected_image, rotated_image, rotation, cv::Size(w*2, h*2));

                cv::imshow("roatated image", rotated_image);
                cv::waitKey(2);

                Rectangle rect(j, i);

                for(int x = rect.u_l.x; x < rect.u_r.x; x++)
                {
                    for(int y = rect.u_l.y; y < rect.l_l.y; y++)
                    {
                        if(rotated_image.at<uchar>(y, x) == 255)
                        {
                            num++;
                        }
                    }
                }

                if(num > best_num)
                {
                    best_theta = theta;
                    best_num = num;
                    best_cent = cent;
                }
            }
        }
    }

    cout << "best num: " << best_num << endl;
    cout << "best theta: " << best_theta << endl;
    cout << "best center: " << best_cent << endl;

    cv::Point2f vertices2f[4];
    cv::RotatedRect rotated_rect(best_cent, cv::Size(dim_w, dim_h), - best_theta);
    rotated_rect.points(vertices2f);  

    for (int i = 0; i < 4; i ++)
    {   
        vertices[i] = vertices2f[i];
        cout << vertices[i] << endl;    
    }
    */

}


Rect lineFitting(cv::Mat image)
{   

    int thresh = 100;

    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using Threshold
    threshold( image, threshold_output, thresh, 255, THRESH_BINARY );
    /// Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );

    Rect best_boundRect;

    for( int i = 0; i < contours.size(); i++ )
        { 
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }


    /// Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC1 );
    
    double area;
    double max_area = 0;

    for( int i = 0; i< contours.size(); i++ )
        {   
            double area = boundRect[i].area();
            if(area > max_area)
            {
                max_area = area;
                best_boundRect = boundRect[i];
            }
            //drawContours( drawing, contours_poly, i, 255, 1, 8, vector<Vec4i>(), 0, Point() );
            
            //circle( drawing, center[i], (int)radius[i], 255, 1, 8, 0 );
        }

    rectangle( drawing, best_boundRect.tl(), best_boundRect.br(), 255, 1, 8, 0 );

    /*
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    waitKey(2);
    */

    /*
    vector<vector<Point> >hull( contours.size() );
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        convexHull( contours[i], hull[i] );
    }
    
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC1 );
    
    for( size_t i = 0; i< contours.size(); i++ )
    {
        //drawContours( drawing, contours, (int)i, color );
        drawContours( drawing, hull, (int)i, 255 );
    }
    
    cout << " Here? " << endl;

    //waitKey(0);
    */
    
    /*
    int thresh = 100;

    Mat canny_output;
    //vector<vector<Point> > contours;
    //vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( image, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    //findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC1 );
    for( int i = 0; i< contours.size(); i++ )
    {
       drawContours( drawing, contours, i, 255, 1, 8, hierarchy, 0, Point() );
    }

    imshow("contour", drawing);
    */

    /*
    int array[drawing.rows][2];

    for (int i = 0; i < drawing.rows; i++)
    {   
        int num = 0;

        for (int j = 0; j < drawing.cols; j++)
        {
            if(drawing.at<Vec3b>(i, j)[0] == 0 && drawing.at<Vec3b>(i, j)[1] == 0 && drawing.at<Vec3b>(i, j)[2] == 0)
            {}
            else
            {
                array[i][num] = j;
                num++;
            }
        }
    }
    */

    /*
    int cols = image.cols;

    array<double, 4> output_line;
    fitLine(canny_output, output_line, CV_DIST_L2, 0, 0.01, 0.01);

    cout << output_line[0] << endl;
    
    double vx = output_line[0];
    double vy = output_line[1];
    double x = output_line[2];
    double y = output_line[3];

    int lefty = (int)((- x * vy / vx) + y);
    int righty = (int)(((cols - x) * vy / vx) + y);
    
    Point poS;
    Point poL;

    poS.x = cols - 1;
    poS.y = righty;

    poL.x = 0;
    poL.y = lefty;

    RNG rng(12345);
    Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );

    line(image, poS, poL, color);
    imshow("image", image);
    waitKey(0);
    */


    return best_boundRect;
}   

Point Rotate(float x, float y, float xm, float ym, float theta)
{
    float rad = theta * PI / 180.0;
    float xr = (x - xm) * cos(rad) + (y - ym) * sin(rad)  + xm;
    float yr = - (x - xm) * sin(rad) + (y - ym) * cos(rad)  + ym;

    Point r (xr, yr);

    return r;
}

double GetDist(Plane::Plane_model plane, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outlier)
{   
    double dist;
    double max_dist = 0;

    double a = plane.a;
    double b = plane.b;
    double c = plane.c;
    double d = plane.d;

    for(int i = 0; i < cloud_outlier->points.size(); i++)
    {
        double x = cloud_outlier->points[i].x;
        double y = cloud_outlier->points[i].y;
        double z = cloud_outlier->points[i].z;

        dist = abs((a * x + b * y + c * z + d) / plane.denominator);

        if(dist > 0.07)
            { }
        else if (dist > max_dist)
            { max_dist = dist; }
    }

    cout << " max_dist : " << max_dist << endl;

    if(max_dist < 0.045 && max_dist > 0.005)
        { return 0.025; }
    else if(max_dist < 0.07 && max_dist > 0.03)
        { return 0.05; }
    else{ return 0; }
    // We need break;
}

void Merge(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> color_vector)
{
    for (int i = 0; i < color_vector.at(0)->points.size(); i++)
    {
        pcl::PointXYZ pt = color_vector.at(0)->points[i];
        merged_cloud->points.push_back(pt);
    }

    for (int i = 0; i < color_vector.at(1)->points.size(); i++)
    {
        pcl::PointXYZ pt = color_vector.at(1)->points[i];
        merged_cloud->points.push_back(pt);
    }
} 

void vizPointCloud(cv::Mat image_RGB, pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud)
{   
    float fx = 612.86083984375;
    float fy = 613.0430908203125;
    float cx = 316.27764892578125; 
    float cy = 250.1717071533203;

    cv::Mat cloud_image = cv::Mat::zeros(image_RGB.rows, image_RGB.cols, CV_8UC1);

    for(int i = 0; i < merged_cloud->points.size(); i++)
    {
        pcl::PointXYZ pt = merged_cloud->points[i];
        float X = pt.x;
        float Y = pt.y;
        float Z = pt.z;

        int x = cy + fy * Y / Z;
        int y = cx + fx * X / Z;

        if( x > 0 && x < image_RGB.rows && y > 0 && y < image_RGB.cols)
            { cloud_image.at<uchar>(x, y) = 255; }       
    }

    cv::imshow("cloud image visualization", cloud_image);
    cv::waitKey(2);
}

pcl::PointCloud<pcl::PointXYZ> makeSyntheticCloud(std::string color_string)
{   

    if(color_string == "red")
    {   
        pcl::PointCloud<pcl::PointXYZ> output_cloud;
        output_cloud = GetRedSynthetic();

        return output_cloud;
    }
    else if(color_string == "orange")
    {
        pcl::PointCloud<pcl::PointXYZ> output_cloud;
        output_cloud = GetOrangeSynthetic();

        return output_cloud;
    }
    else if(color_string == "yellow")
    {
        pcl::PointCloud<pcl::PointXYZ> output_cloud;
        output_cloud = GetYellowSynthetic();

        return output_cloud;        
    }
    else if(color_string == "green")
    {
        pcl::PointCloud<pcl::PointXYZ> output_cloud;
        output_cloud = GetGreenSynthetic();

        return output_cloud;        
    }
    else if(color_string == "blue")
    {   
        pcl::PointCloud<pcl::PointXYZ> output_cloud;
        output_cloud = GetBlueSynthetic();

        return output_cloud; 
    }
    else if(color_string == "purple")
    {   
        pcl::PointCloud<pcl::PointXYZ> output_cloud;
        output_cloud = GetPurpleSynthetic();

        return output_cloud;
    }
    else if(color_string == "brown")
    {
        pcl::PointCloud<pcl::PointXYZ> output_cloud;
        output_cloud = GetBrownSynthetic();

        return output_cloud;
    }

}   

double getDistant(pcl::PointXYZ pt_1, pcl::PointXYZ pt_2)
{   
    double distance = sqrt(pow((pt_1.x - pt_2.x), 2) + pow((pt_1.y - pt_2.y), 2) + pow((pt_1.z - pt_2.z), 2));
    return distance;
}

int getStep(float number)
{
    if(number > 0.02 && number < 0.03)
        { return 1; }
    else if(number > 0.045 && number < 0.055)
        { return 2; }
    else if(number > 0.07 && number < 0.08)
        { return 3; }
    else 
        { return 0; }
}


pcl::PointCloud<pcl::PointXYZ> GetRedSynthetic()
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr red_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    red_cloud = red_vector.at(0);

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    float block_width = getDistant(red_keypoints.at(0), red_keypoints.at(1));
    float block_depth = getDistant(red_keypoints.at(0), red_keypoints.at(3));

    int width_step = getStep(block_width);
    int depth_step = getStep(block_depth);
    int height_step = getStep(red_block_height);

    cout << "width_step: " << width_step << ", " << "depth_step: " << depth_step << ", " << "height_step: " << height_step <<  endl;

    if(width_step * depth_step * height_step == 4)
    {   
        cout << " We are here~ " << endl;

        std::vector<float> width_vector = std::vector<float> (3);
        width_vector.at(0) = red_keypoints.at(1).x - red_keypoints.at(0).x;
        width_vector.at(1) = red_keypoints.at(1).y - red_keypoints.at(0).y;
        width_vector.at(2) = red_keypoints.at(1).z - red_keypoints.at(0).z;

        float norm = sqrt(pow(width_vector.at(0), 2) + pow(width_vector.at(1),2) + pow(width_vector.at(2), 2));

        width_vector.at(0) = width_vector.at(0) / norm;
        width_vector.at(1) = width_vector.at(1) / norm;
        width_vector.at(2) = width_vector.at(2) / norm;

        std::vector<float> depth_vector = std::vector<float> (3);
        depth_vector.at(0) = red_keypoints.at(3).x - red_keypoints.at(0).x;
        depth_vector.at(1) = red_keypoints.at(3).y - red_keypoints.at(0).y;
        depth_vector.at(2) = red_keypoints.at(3).z - red_keypoints.at(0).z;

        norm = sqrt(pow(depth_vector.at(0), 2) + pow(depth_vector.at(1),2) + pow(depth_vector.at(2), 2));

        depth_vector.at(0) = depth_vector.at(0) / norm;
        depth_vector.at(1) = depth_vector.at(1) / norm;
        depth_vector.at(2) = depth_vector.at(2) / norm;
        
        std::vector<float> height_vector = std::vector<float> (3);
        height_vector.at(0) = unit_vector.at(0);
        height_vector.at(1) = unit_vector.at(1);
        height_vector.at(2) = unit_vector.at(2);

        int output_array[width_step][depth_step][height_step] = {};

       for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                 output_array[j][k][i] = 0;
                }
            }
        }

        for (int n = 0; n < red_cloud->points.size(); n++)
        {
            std::vector<int> index = std::vector<int> (3);

            pcl::PointXYZ pt;
            pt.x = red_cloud->points[n].x;
            pt.y = red_cloud->points[n].y;
            pt.z = red_cloud->points[n].z;


            float min_dist = 100;
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(pt.x == 0 && pt.y == 0 && pt.z == 0)
                        {}
                        else
                        {   pcl::PointXYZ point;
                            point.x = red_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = red_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = red_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2;

                            double dist = getDistant(pt, point);

                            if(dist < min_dist)
                            {
                                min_dist = dist;
                                index.at(0) = j;
                                index.at(1) = k;
                                index.at(2) = i;
                            }
                        }
                    }
                }
            }

            output_array[index.at(0)][index.at(1)][index.at(2)] += 1;
        }

        if(height_step == 2)
        {   
            int max_value = 0;
            std::vector<int> max_index = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] > max_value)
                        {   
                            max_value = output_array[j][k][i];
                            max_index.at(0) = j;
                            max_index.at(1) = k;
                            max_index.at(2) = i;
                        }
                    }
                }
            }

            int max_value_2 = 0;
            std::vector<int> max_index_2 = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                        {}
                        else
                        {
                            if(output_array[j][k][i] > max_value_2)
                            {   
                                max_value_2 = output_array[j][k][i];
                                max_index_2.at(0) = j;
                                max_index_2.at(1) = k;
                                max_index_2.at(2) = i;
                            }
                        }
                    }
                }
            }

            
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if((j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2)) || (j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2)))
                        {   
                            pcl::PointXYZ point;
                            point.x = red_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = red_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = red_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
  
                            output_cloud.points.push_back(point);

                            if(i == 1)
                            {   
                                pcl::PointXYZ bottom_point;
                                bottom_point.x = red_keypoints.at(0).x + height_vector.at(0)*(0.025)*(i - 1) + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                                bottom_point.y = red_keypoints.at(0).y + height_vector.at(1)*(0.025)*(i - 1) + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                                bottom_point.z = red_keypoints.at(0).z + height_vector.at(2)*(0.025)*(i - 1) + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
    
                                output_cloud.points.push_back(bottom_point);
                            }
                        }
                    }
                }
            }

            if(output_cloud.points.size() == 3)
                { pcl::io::savePLYFileASCII ("red_output_cloud.ply", output_cloud); return output_cloud; }
        }
        else
        {   
            int min_value = 10000;
            std::vector<int> min_index = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] < min_value)
                        {   
                            min_value = output_array[j][k][i];
                            min_index.at(0) = j;
                            min_index.at(1) = k;
                            min_index.at(2) = i;
                        }
                    }
                }
            }
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(min_index.at(0) == j && min_index.at(1) == k && min_index.at(2) == i)
                        {}
                        else
                        {   

                            pcl::PointXYZ point;
                            point.x = red_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = red_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = red_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2;

                            output_cloud.points.push_back(point); 
                        }
                    }
                }
            }
            if(output_cloud.points.size() == 3)
                { pcl::io::savePLYFileASCII ("red_output_cloud.ply", output_cloud); return output_cloud; }
        } 
    }
}

pcl::PointCloud<pcl::PointXYZ> GetOrangeSynthetic()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr orange_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    orange_cloud = orange_vector.at(0);

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    float block_width = getDistant(orange_keypoints.at(0), orange_keypoints.at(1));
    float block_depth = getDistant(orange_keypoints.at(0), orange_keypoints.at(3));

    int width_step = getStep(block_width);
    int depth_step = getStep(block_depth);
    int height_step = getStep(orange_block_height);

    cout << "width_step: " << width_step << ", " << "depth_step: " << depth_step << ", " << "height_step: " << height_step <<  endl;

    if(width_step * depth_step * height_step == 6)
    {   
        cout << " I'm Orange~ " << endl;

        std::vector<float> width_vector = std::vector<float> (3);
        width_vector.at(0) = orange_keypoints.at(1).x - orange_keypoints.at(0).x;
        width_vector.at(1) = orange_keypoints.at(1).y - orange_keypoints.at(0).y;
        width_vector.at(2) = orange_keypoints.at(1).z - orange_keypoints.at(0).z;

        float norm = sqrt(pow(width_vector.at(0), 2) + pow(width_vector.at(1),2) + pow(width_vector.at(2), 2));

        width_vector.at(0) = width_vector.at(0) / norm;
        width_vector.at(1) = width_vector.at(1) / norm;
        width_vector.at(2) = width_vector.at(2) / norm;

        std::vector<float> depth_vector = std::vector<float> (3);
        depth_vector.at(0) = orange_keypoints.at(3).x - orange_keypoints.at(0).x;
        depth_vector.at(1) = orange_keypoints.at(3).y - orange_keypoints.at(0).y;
        depth_vector.at(2) = orange_keypoints.at(3).z - orange_keypoints.at(0).z;

        norm = sqrt(pow(depth_vector.at(0), 2) + pow(depth_vector.at(1),2) + pow(depth_vector.at(2), 2));

        depth_vector.at(0) = depth_vector.at(0) / norm;
        depth_vector.at(1) = depth_vector.at(1) / norm;
        depth_vector.at(2) = depth_vector.at(2) / norm;

        std::vector<float> height_vector = std::vector<float> (3);
        height_vector.at(0) = unit_vector.at(0);
        height_vector.at(1) = unit_vector.at(1);
        height_vector.at(2) = unit_vector.at(2); // It must be same at any color


        double dist_array[width_step][depth_step][height_step] = {};
        int output_array[width_step][depth_step][height_step] = {};

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    dist_array[j][k][i] = 0;
                    output_array[j][k][i] = 0;
                }
            }
        }

        std::vector<int> index = std::vector<int> (3);

        for (int n = 0; n < orange_cloud->points.size(); n++)
        {                   
            pcl::PointXYZ pt;
            pt.x = orange_cloud->points[n].x;
            pt.y = orange_cloud->points[n].y;
            pt.z = orange_cloud->points[n].z;


            if(pt.x == 0 && pt.y == 0 && pt.z == 0)
            { }
            else{
                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            pcl::PointXYZ point;
                            point.x = orange_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = orange_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = orange_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            double dist = getDistant(pt, point);

                            dist_array[j][k][i] = dist;
                        }
                    }
                }

                double min_dist = 100;
                int min_i;
                int min_j;
                int min_k;

                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            if(dist_array[j][k][i] <= min_dist)
                            { min_dist = dist_array[j][k][i]; min_i = i; min_j = j; min_k = k;} 
                        }
                    }
                }
                output_array[min_j][min_k][min_i] += 1;
            } 
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    cout << "value : " << output_array[j][k][i] << endl;
                }
            }
        }
        if(height_step == 2)
        {
            int max_value = 0;
            std::vector<int> max_index = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] > max_value)
                        {   
                            max_value = output_array[j][k][i];
                            max_index.at(0) = j;
                            max_index.at(1) = k;
                            max_index.at(2) = i;
                        }
                    }
                }
            }

            int max_value_2 = 0;
            std::vector<int> max_index_2 = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                        {}
                        else
                        {
                            if(output_array[j][k][i] > max_value_2)
                            {   
                                max_value_2 = output_array[j][k][i];
                                max_index_2.at(0) = j;
                                max_index_2.at(1) = k;
                                max_index_2.at(2) = i;
                            }
                        }
                    }
                }
            }

            int max_value_3 = 0;
            std::vector<int> max_index_3 = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                        {}
                        else if(j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2))
                        {}
                        else{

                            if(output_array[j][k][i] > max_value_3)
                            {   
                                max_value_3 = output_array[j][k][i];
                                max_index_3.at(0) = j;
                                max_index_3.at(1) = k;
                                max_index_3.at(2) = i;
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if((j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2)) || (j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2)) || (j == max_index_3.at(0) && k == max_index_3.at(1) && i == max_index_3.at(2)))
                        {   
                            pcl::PointXYZ point;
                            point.x = orange_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = orange_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = orange_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
  
                            output_cloud.points.push_back(point);

                            if(i == 1)
                            {   
                                pcl::PointXYZ bottom_point;
                                bottom_point.x = orange_keypoints.at(0).x + height_vector.at(0)*(0.025)*(i - 1) + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                                bottom_point.y = orange_keypoints.at(0).y + height_vector.at(1)*(0.025)*(i - 1) + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                                bottom_point.z = orange_keypoints.at(0).z + height_vector.at(2)*(0.025)*(i - 1) + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
    
                                output_cloud.points.push_back(bottom_point);
                            }
                        }
                    }
                }
            }

            if(output_cloud.points.size() == 4)
                { pcl::io::savePLYFileASCII ("orange_output_cloud.ply", output_cloud); return output_cloud; }
        }
        else
        {   
            int min_value = 100000;
            std::vector<int> min_index = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] < min_value)
                        {   
                            min_value = output_array[j][k][i];
                            min_index.at(0) = j;
                            min_index.at(1) = k;
                            min_index.at(2) = i;
                        }
                    }
                }
            }

            int min_value_2 = 100000;
            std::vector<int> min_index_2 = std::vector<int> (3);
           for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == min_index.at(0) && k == min_index.at(1) && i == min_index.at(2))
                        {}
                        else
                        {
                            if(output_array[j][k][i] < min_value_2)
                            {   
                                min_value_2 = output_array[j][k][i];
                                min_index_2.at(0) = j;
                                min_index_2.at(1) = k;
                                min_index_2.at(2) = i;
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(min_index.at(0) == j && min_index.at(1) == k && min_index.at(2) == i)
                        {}
                        else if(min_index_2.at(0) == j && min_index_2.at(1) == k && min_index_2.at(2) == i)
                        {}
                        else
                        { 
                            pcl::PointXYZ point;
                            point.x = orange_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = orange_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = orange_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
  
                            output_cloud.points.push_back(point); 
                        }
                    }
                }
            }
            if(output_cloud.points.size() == 4)
                { pcl::io::savePLYFileASCII ("orange_output_cloud.ply", output_cloud); return output_cloud; }
        }
    }      
}

pcl::PointCloud<pcl::PointXYZ> GetYellowSynthetic()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr yellow_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    yellow_cloud = yellow_vector.at(0);

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    float block_width = getDistant(yellow_keypoints.at(0), yellow_keypoints.at(1));
    float block_depth = getDistant(yellow_keypoints.at(0), yellow_keypoints.at(3));

    int width_step = getStep(block_width);
    int depth_step = getStep(block_depth);
    int height_step = getStep(yellow_block_height);

    cout << "width_step: " << width_step << ", " << "depth_step: " << depth_step << ", " << "height_step: " << height_step <<  endl;

    if(width_step * depth_step * height_step == 6)
    {   
        cout << " I'm Yellow~ " << endl;

        std::vector<float> width_vector = std::vector<float> (3);
        width_vector.at(0) = yellow_keypoints.at(1).x - yellow_keypoints.at(0).x;
        width_vector.at(1) = yellow_keypoints.at(1).y - yellow_keypoints.at(0).y;
        width_vector.at(2) = yellow_keypoints.at(1).z - yellow_keypoints.at(0).z;

        float norm = sqrt(pow(width_vector.at(0), 2) + pow(width_vector.at(1),2) + pow(width_vector.at(2), 2));

        width_vector.at(0) = width_vector.at(0) / norm;
        width_vector.at(1) = width_vector.at(1) / norm;
        width_vector.at(2) = width_vector.at(2) / norm;

        std::vector<float> depth_vector = std::vector<float> (3);
        depth_vector.at(0) = yellow_keypoints.at(3).x - yellow_keypoints.at(0).x;
        depth_vector.at(1) = yellow_keypoints.at(3).y - yellow_keypoints.at(0).y;
        depth_vector.at(2) = yellow_keypoints.at(3).z - yellow_keypoints.at(0).z;

        norm = sqrt(pow(depth_vector.at(0), 2) + pow(depth_vector.at(1),2) + pow(depth_vector.at(2), 2));

        depth_vector.at(0) = depth_vector.at(0) / norm;
        depth_vector.at(1) = depth_vector.at(1) / norm;
        depth_vector.at(2) = depth_vector.at(2) / norm;

        std::vector<float> height_vector = std::vector<float> (3);
        height_vector.at(0) = unit_vector.at(0);
        height_vector.at(1) = unit_vector.at(1);
        height_vector.at(2) = unit_vector.at(2); // It must be same at any color


        double dist_array[width_step][depth_step][height_step] = {};
        int output_array[width_step][depth_step][height_step] = {};

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    dist_array[j][k][i] = 0;
                    output_array[j][k][i] = 0;
                }
            }
        }

        std::vector<int> index = std::vector<int> (3);

        for (int n = 0; n < yellow_cloud->points.size(); n++)
        {                   
            pcl::PointXYZ pt;
            pt.x = yellow_cloud->points[n].x;
            pt.y = yellow_cloud->points[n].y;
            pt.z = yellow_cloud->points[n].z;


            if(pt.x == 0 && pt.y == 0 && pt.z == 0)
            { }
            else{
                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            pcl::PointXYZ point;
                            point.x = yellow_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = yellow_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = yellow_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            double dist = getDistant(pt, point);

                            dist_array[j][k][i] = dist;
                        }
                    }
                }

                double min_dist = 100;
                int min_i;
                int min_j;
                int min_k;

                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            if(dist_array[j][k][i] <= min_dist)
                            { min_dist = dist_array[j][k][i]; min_i = i; min_j = j; min_k = k;} 
                        }
                    }
                }
                output_array[min_j][min_k][min_i] += 1;
            } 
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    cout << "value : " << output_array[j][k][i] << endl;
                }
            }
        }

        if(height_step == 2)
        {
            int max_value = 0;
            std::vector<int> max_index = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] > max_value)
                        {   
                            max_value = output_array[j][k][i];
                            max_index.at(0) = j;
                            max_index.at(1) = k;
                            max_index.at(2) = i;
                        }
                    }
                }
            }

            int max_value_2 = 0;
            std::vector<int> max_index_2 = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                        {}
                        else
                        {
                            if(output_array[j][k][i] > max_value_2)
                            {   
                                max_value_2 = output_array[j][k][i];
                                max_index_2.at(0) = j;
                                max_index_2.at(1) = k;
                                max_index_2.at(2) = i;
                            }
                        }
                    }
                }
            }

            int max_value_3 = 0;
            std::vector<int> max_index_3 = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                        {}
                        else if(j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2))
                        {}
                        else{

                            if(output_array[j][k][i] > max_value_3)
                            {   
                                max_value_3 = output_array[j][k][i];
                                max_index_3.at(0) = j;
                                max_index_3.at(1) = k;
                                max_index_3.at(2) = i;
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if((j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2)) || (j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2)) || (j == max_index_3.at(0) && k == max_index_3.at(1) && i == max_index_3.at(2)))
                        {   
                            pcl::PointXYZ point;
                            point.x = yellow_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = yellow_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = yellow_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
  
                            output_cloud.points.push_back(point);

                            if(i == 1)
                            {   
                                pcl::PointXYZ bottom_point;
                                bottom_point.x = yellow_keypoints.at(0).x + height_vector.at(0)*(0.025)*(i - 1) + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                                bottom_point.y = yellow_keypoints.at(0).y + height_vector.at(1)*(0.025)*(i - 1) + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                                bottom_point.z = yellow_keypoints.at(0).z + height_vector.at(2)*(0.025)*(i - 1) + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
    
                                output_cloud.points.push_back(bottom_point);
                            }
                        }
                    }
                }
            }
            if(output_cloud.points.size() == 4)
                { pcl::io::savePLYFileASCII ("yellow_output_cloud.ply", output_cloud); return output_cloud;}
        }
        else 
        {   
            int min_value = 100000;
            std::vector<int> min_index = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] < min_value)
                        {   
                            min_value = output_array[j][k][i];
                            min_index.at(0) = j;
                            min_index.at(1) = k;
                            min_index.at(2) = i;
                        }
                    }
                }
            }

            int min_value_2 = 100000;
            std::vector<int> min_index_2 = std::vector<int> (3);
           for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == min_index.at(0) && k == min_index.at(1) && i == min_index.at(2))
                        {}
                        else
                        {
                            if(output_array[j][k][i] < min_value_2)
                            {   
                                min_value_2 = output_array[j][k][i];
                                min_index_2.at(0) = j;
                                min_index_2.at(1) = k;
                                min_index_2.at(2) = i;
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(min_index.at(0) == j && min_index.at(1) == k && min_index.at(2) == i)
                        {}
                        else if(min_index_2.at(0) == j && min_index_2.at(1) == k && min_index_2.at(2) == i)
                        {}
                        else
                        { 
                            pcl::PointXYZ point;
                            point.x = yellow_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = yellow_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = yellow_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
  
                            output_cloud.points.push_back(point); 
                        }
                    }
                }
            }
            if(output_cloud.points.size() == 4)
                { pcl::io::savePLYFileASCII ("yellow_output_cloud.ply", output_cloud); return output_cloud;}
        }
    }      
}

pcl::PointCloud<pcl::PointXYZ> GetGreenSynthetic()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr green_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    green_cloud = green_vector.at(0);

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    float block_width = getDistant(green_keypoints.at(0), green_keypoints.at(1));
    float block_depth = getDistant(green_keypoints.at(0), green_keypoints.at(3));

    int width_step = getStep(block_width);
    int depth_step = getStep(block_depth);
    int height_step = getStep(green_block_height);

    cout << "width_step: " << width_step << ", " << "depth_step: " << depth_step << ", " << "height_step: " << height_step <<  endl;

    if(width_step * depth_step * height_step == 6)
    {   
        cout << " I'm Green~ " << endl;

        std::vector<float> width_vector = std::vector<float> (3);
        width_vector.at(0) = green_keypoints.at(1).x - green_keypoints.at(0).x;
        width_vector.at(1) = green_keypoints.at(1).y - green_keypoints.at(0).y;
        width_vector.at(2) = green_keypoints.at(1).z - green_keypoints.at(0).z;

        float norm = sqrt(pow(width_vector.at(0), 2) + pow(width_vector.at(1),2) + pow(width_vector.at(2), 2));

        width_vector.at(0) = width_vector.at(0) / norm;
        width_vector.at(1) = width_vector.at(1) / norm;
        width_vector.at(2) = width_vector.at(2) / norm;

        std::vector<float> depth_vector = std::vector<float> (3);
        depth_vector.at(0) = green_keypoints.at(3).x - green_keypoints.at(0).x;
        depth_vector.at(1) = green_keypoints.at(3).y - green_keypoints.at(0).y;
        depth_vector.at(2) = green_keypoints.at(3).z - green_keypoints.at(0).z;

        norm = sqrt(pow(depth_vector.at(0), 2) + pow(depth_vector.at(1),2) + pow(depth_vector.at(2), 2));

        depth_vector.at(0) = depth_vector.at(0) / norm;
        depth_vector.at(1) = depth_vector.at(1) / norm;
        depth_vector.at(2) = depth_vector.at(2) / norm;

        std::vector<float> height_vector = std::vector<float> (3);
        height_vector.at(0) = unit_vector.at(0);
        height_vector.at(1) = unit_vector.at(1);
        height_vector.at(2) = unit_vector.at(2); // It must be same at any color


        double dist_array[width_step][depth_step][height_step] = {};
        int output_array[width_step][depth_step][height_step] = {};

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    dist_array[j][k][i] = 0;
                    output_array[j][k][i] = 0;
                }
            }
        }

        std::vector<int> index = std::vector<int> (3);

        for (int n = 0; n < green_cloud->points.size(); n++)
        {                   
            pcl::PointXYZ pt;
            pt.x = green_cloud->points[n].x;
            pt.y = green_cloud->points[n].y;
            pt.z = green_cloud->points[n].z;


            if(pt.x == 0 && pt.y == 0 && pt.z == 0)
            { }
            else{
                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            pcl::PointXYZ point;
                            point.x = green_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = green_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = green_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            double dist = getDistant(pt, point);

                            dist_array[j][k][i] = dist;
                        }
                    }
                }

                double min_dist = 100;
                int min_i;
                int min_j;
                int min_k;

                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            if(dist_array[j][k][i] <= min_dist)
                            { min_dist = dist_array[j][k][i]; min_i = i; min_j = j; min_k = k;} 
                        }
                    }
                }
                output_array[min_j][min_k][min_i] += 1;
            } 
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    cout << "value : " << output_array[j][k][i] << endl;
                }
            }
        }


        if(height_step == 2)
        {
            int max_value = 0;
            std::vector<int> max_index = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] > max_value)
                        {   
                            max_value = output_array[j][k][i];
                            max_index.at(0) = j;
                            max_index.at(1) = k;
                            max_index.at(2) = i;
                        }
                    }
                }
            }

            int max_value_2 = 0;
            std::vector<int> max_index_2 = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                        {}
                        else
                        {
                            if(output_array[j][k][i] > max_value_2)
                            {   
                                max_value_2 = output_array[j][k][i];
                                max_index_2.at(0) = j;
                                max_index_2.at(1) = k;
                                max_index_2.at(2) = i;
                            }
                        }
                    }
                }
            }

            int max_value_3 = 0;
            std::vector<int> max_index_3 = std::vector<int> (3);
            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                        {}
                        else if(j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2))
                        {}
                        else{

                            if(output_array[j][k][i] > max_value_3)
                            {   
                                max_value_3 = output_array[j][k][i];
                                max_index_3.at(0) = j;
                                max_index_3.at(1) = k;
                                max_index_3.at(2) = i;
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if((j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2)) || (j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2)) || (j == max_index_3.at(0) && k == max_index_3.at(1) && i == max_index_3.at(2)))
                        {   
                            pcl::PointXYZ point;
                            point.x = green_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = green_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = green_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
  
                            output_cloud.points.push_back(point);

                            if(j == 1 || k == 1)
                            {   
                                pcl::PointXYZ bottom_point;
                                bottom_point.x = green_keypoints.at(0).x + height_vector.at(0)*(0.025)*(i - 1) + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                                bottom_point.y = green_keypoints.at(0).y + height_vector.at(1)*(0.025)*(i - 1) + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                                bottom_point.z = green_keypoints.at(0).z + height_vector.at(2)*(0.025)*(i - 1) + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
    
                                output_cloud.points.push_back(bottom_point);
                            }
                        }
                    }
                }
            }
            if(output_cloud.points.size() == 4)
                { pcl::io::savePLYFileASCII ("green_output_cloud.ply", output_cloud); return output_cloud; } 
        }
        else
        {   
            int min_value = 100000;
            std::vector<int> min_index = std::vector<int> (3);
           for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(output_array[j][k][i] < min_value)
                        {   
                            min_value = output_array[j][k][i];
                            min_index.at(0) = j;
                            min_index.at(1) = k;
                            min_index.at(2) = i;
                        }
                    }
                }
            }

            int min_value_2 = 100000;
            std::vector<int> min_index_2 = std::vector<int> (3);
           for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(j == min_index.at(0) && k == min_index.at(1) && i == min_index.at(2))
                        {}
                        else
                        {
                            if(output_array[j][k][i] < min_value_2)
                            {   
                                min_value_2 = output_array[j][k][i];
                                min_index_2.at(0) = j;
                                min_index_2.at(1) = k;
                                min_index_2.at(2) = i;
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < height_step; i++)
            {
                for(int j = 0; j < width_step; j++)
                {
                    for(int k = 0; k < depth_step; k++)
                    {   
                        if(min_index.at(0) == j && min_index.at(1) == k && min_index.at(2) == i)
                        {}
                        else if(min_index_2.at(0) == j && min_index_2.at(1) == k && min_index_2.at(2) == i)
                        {}
                        else
                        { 
                            pcl::PointXYZ point;
                            point.x = green_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = green_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = green_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 
  
                            output_cloud.points.push_back(point); 
                        }
                    }
                }
            }
            if(output_cloud.points.size() == 4)
                { pcl::io::savePLYFileASCII ("green_output_cloud.ply", output_cloud); return output_cloud; } 
        }    
    }      
}

pcl::PointCloud<pcl::PointXYZ> GetBlueSynthetic()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr blue_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    blue_cloud = blue_vector.at(0);

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    float block_width = getDistant(blue_keypoints.at(0), blue_keypoints.at(1));
    float block_depth = getDistant(blue_keypoints.at(0), blue_keypoints.at(3));

    int width_step = getStep(block_width);
    int depth_step = getStep(block_depth);
    int height_step = getStep(blue_block_height);

    cout << "width_step: " << width_step << ", " << "depth_step: " << depth_step << ", " << "height_step: " << height_step <<  endl;

    if(width_step * depth_step * height_step == 8)
    {   
        cout << " I'm Blue~ " << endl;

        std::vector<float> width_vector = std::vector<float> (3);
        width_vector.at(0) = blue_keypoints.at(1).x - blue_keypoints.at(0).x;
        width_vector.at(1) = blue_keypoints.at(1).y - blue_keypoints.at(0).y;
        width_vector.at(2) = blue_keypoints.at(1).z - blue_keypoints.at(0).z;

        float norm = sqrt(pow(width_vector.at(0), 2) + pow(width_vector.at(1),2) + pow(width_vector.at(2), 2));

        width_vector.at(0) = width_vector.at(0) / norm;
        width_vector.at(1) = width_vector.at(1) / norm;
        width_vector.at(2) = width_vector.at(2) / norm;

        std::vector<float> depth_vector = std::vector<float> (3);
        depth_vector.at(0) = blue_keypoints.at(3).x - blue_keypoints.at(0).x;
        depth_vector.at(1) = blue_keypoints.at(3).y - blue_keypoints.at(0).y;
        depth_vector.at(2) = blue_keypoints.at(3).z - blue_keypoints.at(0).z;

        norm = sqrt(pow(depth_vector.at(0), 2) + pow(depth_vector.at(1),2) + pow(depth_vector.at(2), 2));

        depth_vector.at(0) = depth_vector.at(0) / norm;
        depth_vector.at(1) = depth_vector.at(1) / norm;
        depth_vector.at(2) = depth_vector.at(2) / norm;

        std::vector<float> height_vector = std::vector<float> (3);
        height_vector.at(0) = unit_vector.at(0);
        height_vector.at(1) = unit_vector.at(1);
        height_vector.at(2) = unit_vector.at(2); // It must be same at any color


        double dist_array[width_step][depth_step][height_step] = {};
        int output_array[width_step][depth_step][height_step] = {};

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    dist_array[j][k][i] = 0;
                    output_array[j][k][i] = 0;
                }
            }
        }

        std::vector<int> index = std::vector<int> (3);

        for (int n = 0; n < blue_cloud->points.size(); n++)
        {                   
            pcl::PointXYZ pt;
            pt.x = blue_cloud->points[n].x;
            pt.y = blue_cloud->points[n].y;
            pt.z = blue_cloud->points[n].z;


            if(pt.x == 0 && pt.y == 0 && pt.z == 0)
            { }
            else{
                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            pcl::PointXYZ point;
                            point.x = blue_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = blue_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = blue_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            double dist = getDistant(pt, point);

                            dist_array[j][k][i] = dist;
                        }
                    }
                }

                double min_dist = 100;
                int min_i;
                int min_j;
                int min_k;

                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            if(dist_array[j][k][i] <= min_dist)
                            { min_dist = dist_array[j][k][i]; min_i = i; min_j = j; min_k = k;} 
                        }
                    }
                }
                output_array[min_j][min_k][min_i] += 1;
            } 
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    cout << "value : " << output_array[j][k][i] << endl;
                }
            }
        }

        int max_value = 0;
        std::vector<int> max_index = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(output_array[j][k][i] > max_value)
                    {   
                        max_value = output_array[j][k][i];
                        max_index.at(0) = j;
                        max_index.at(1) = k;
                        max_index.at(2) = i;
                    }
                }
            }
        }

        int max_value_2 = 0;
        std::vector<int> max_index_2 = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                    {}
                    else
                    {
                        if(output_array[j][k][i] > max_value_2)
                        {   
                            max_value_2 = output_array[j][k][i];
                            max_index_2.at(0) = j;
                            max_index_2.at(1) = k;
                            max_index_2.at(2) = i;
                        }
                    }
                }
            }
        }

        int max_value_3 = 0;
        std::vector<int> max_index_3 = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                    {}
                    else if(j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2))
                    {}
                    else{

                        if(output_array[j][k][i] > max_value_3)
                        {   
                            max_value_3 = output_array[j][k][i];
                            max_index_3.at(0) = j;
                            max_index_3.at(1) = k;
                            max_index_3.at(2) = i;
                        }
                    }
                }
            }
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if((j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2)) || (j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2)) || (j == max_index_3.at(0) && k == max_index_3.at(1) && i == max_index_3.at(2)))
                    {   
                        pcl::PointXYZ point;
                        point.x = blue_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                        point.y = blue_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                        point.z = blue_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                        output_cloud.points.push_back(point);

                        if(i == 1)
                        {   
                            pcl::PointXYZ bottom_point;
                            bottom_point.x = blue_keypoints.at(0).x + height_vector.at(0)*(0.025)*(i - 1) + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            bottom_point.y = blue_keypoints.at(0).y + height_vector.at(1)*(0.025)*(i - 1) + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            bottom_point.z = blue_keypoints.at(0).z + height_vector.at(2)*(0.025)*(i - 1) + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            output_cloud.points.push_back(bottom_point);
                        }
                    }
                }
            }
        }
        if(output_cloud.points.size() == 4)
            { pcl::io::savePLYFileASCII ("blue_output_cloud.ply", output_cloud); return output_cloud; }
    }      
}

pcl::PointCloud<pcl::PointXYZ> GetPurpleSynthetic()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr purple_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    purple_cloud = purple_vector.at(0);

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    float block_width = getDistant(purple_keypoints.at(0), purple_keypoints.at(1));
    float block_depth = getDistant(purple_keypoints.at(0), purple_keypoints.at(3));

    int width_step = getStep(block_width);
    int depth_step = getStep(block_depth);
    int height_step = getStep(purple_block_height);

    cout << "width_step: " << width_step << ", " << "depth_step: " << depth_step << ", " << "height_step: " << height_step <<  endl;

    if(width_step * depth_step * height_step == 8)
    {   
        cout << " I'm Purple~ " << endl;

        std::vector<float> width_vector = std::vector<float> (3);
        width_vector.at(0) = purple_keypoints.at(1).x - purple_keypoints.at(0).x;
        width_vector.at(1) = purple_keypoints.at(1).y - purple_keypoints.at(0).y;
        width_vector.at(2) = purple_keypoints.at(1).z - purple_keypoints.at(0).z;

        float norm = sqrt(pow(width_vector.at(0), 2) + pow(width_vector.at(1),2) + pow(width_vector.at(2), 2));

        width_vector.at(0) = width_vector.at(0) / norm;
        width_vector.at(1) = width_vector.at(1) / norm;
        width_vector.at(2) = width_vector.at(2) / norm;

        std::vector<float> depth_vector = std::vector<float> (3);
        depth_vector.at(0) = purple_keypoints.at(3).x - purple_keypoints.at(0).x;
        depth_vector.at(1) = purple_keypoints.at(3).y - purple_keypoints.at(0).y;
        depth_vector.at(2) = purple_keypoints.at(3).z - purple_keypoints.at(0).z;

        norm = sqrt(pow(depth_vector.at(0), 2) + pow(depth_vector.at(1),2) + pow(depth_vector.at(2), 2));

        depth_vector.at(0) = depth_vector.at(0) / norm;
        depth_vector.at(1) = depth_vector.at(1) / norm;
        depth_vector.at(2) = depth_vector.at(2) / norm;

        std::vector<float> height_vector = std::vector<float> (3);
        height_vector.at(0) = unit_vector.at(0);
        height_vector.at(1) = unit_vector.at(1);
        height_vector.at(2) = unit_vector.at(2); // It must be same at any color

        double dist_array[width_step][depth_step][height_step] = {};
        int output_array[width_step][depth_step][height_step] = {};

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    dist_array[j][k][i] = 0;
                    output_array[j][k][i] = 0;  
                }
            }
        }


        for (int n = 0; n < purple_cloud->points.size(); n++)
        {                   
            pcl::PointXYZ pt;
            pt.x = purple_cloud->points[n].x;
            pt.y = purple_cloud->points[n].y;
            pt.z = purple_cloud->points[n].z;


            if(pt.x == 0 && pt.y == 0 && pt.z == 0)
            { }
            else{
                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            pcl::PointXYZ point;
                            point.x = purple_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = purple_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = purple_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            double dist = getDistant(pt, point);

                            dist_array[j][k][i] = dist;
                        }
                    }
                }

                double min_dist = 100;
                int min_i;
                int min_j;
                int min_k;

                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            if(dist_array[j][k][i] <= min_dist)
                            { min_dist = dist_array[j][k][i]; min_i = i; min_j = j; min_k = k;} 
                        }
                    }
                }
                output_array[min_j][min_k][min_i] += 1;
            } 

        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    cout << "value : " << output_array[j][k][i] << endl;
                }
            }
        }

        int max_value = 0;
        std::vector<int> max_index = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(output_array[j][k][i] > max_value)
                    {   
                        max_value = output_array[j][k][i];
                        max_index.at(0) = j;
                        max_index.at(1) = k;
                        max_index.at(2) = i;
                    }
                }
            }
        }

        int max_value_2 = 0;
        std::vector<int> max_index_2 = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                    {}
                    else
                    {
                        if(output_array[j][k][i] > max_value_2)
                        {   
                            max_value_2 = output_array[j][k][i];
                            max_index_2.at(0) = j;
                            max_index_2.at(1) = k;
                            max_index_2.at(2) = i;
                        }
                    }
                }
            }
        }

        int max_value_3 = 0;
        std::vector<int> max_index_3 = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                    {}
                    else if(j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2))
                    {}
                    else{

                        if(output_array[j][k][i] > max_value_3)
                        {   
                            max_value_3 = output_array[j][k][i];
                            max_index_3.at(0) = j;
                            max_index_3.at(1) = k;
                            max_index_3.at(2) = i;
                        }
                    }
                }
            }
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if((j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2)) || (j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2)) || (j == max_index_3.at(0) && k == max_index_3.at(1) && i == max_index_3.at(2)))
                    {   
                        pcl::PointXYZ point;
                        point.x = purple_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                        point.y = purple_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                        point.z = purple_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                        output_cloud.points.push_back(point);

                        if(i == 1)
                        {   
                            pcl::PointXYZ bottom_point;
                            bottom_point.x = purple_keypoints.at(0).x + height_vector.at(0)*(0.025)*(i - 1) + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            bottom_point.y = purple_keypoints.at(0).y + height_vector.at(1)*(0.025)*(i - 1) + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            bottom_point.z = purple_keypoints.at(0).z + height_vector.at(2)*(0.025)*(i - 1) + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            output_cloud.points.push_back(bottom_point);
                        }
                    }
                }
            }
        }
        
        if(output_cloud.points.size() == 4)
            { pcl::io::savePLYFileASCII ("purple_output_cloud.ply", output_cloud); return output_cloud; }
    }      
}

pcl::PointCloud<pcl::PointXYZ> GetBrownSynthetic()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr brown_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    brown_cloud = brown_vector.at(0);

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    float block_width = getDistant(brown_keypoints.at(0), brown_keypoints.at(1));
    float block_depth = getDistant(brown_keypoints.at(0), brown_keypoints.at(3));

    int width_step = getStep(block_width);
    int depth_step = getStep(block_depth);
    int height_step = getStep(brown_block_height);

    cout << "width_step: " << width_step << ", " << "depth_step: " << depth_step << ", " << "height_step: " << height_step <<  endl;

    if(width_step * depth_step * height_step == 8)
    {   
        cout << " I'm Brown~ " << endl;

        std::vector<float> width_vector = std::vector<float> (3);
        width_vector.at(0) = brown_keypoints.at(1).x - brown_keypoints.at(0).x;
        width_vector.at(1) = brown_keypoints.at(1).y - brown_keypoints.at(0).y;
        width_vector.at(2) = brown_keypoints.at(1).z - brown_keypoints.at(0).z;

        float norm = sqrt(pow(width_vector.at(0), 2) + pow(width_vector.at(1),2) + pow(width_vector.at(2), 2));

        width_vector.at(0) = width_vector.at(0) / norm;
        width_vector.at(1) = width_vector.at(1) / norm;
        width_vector.at(2) = width_vector.at(2) / norm;

        std::vector<float> depth_vector = std::vector<float> (3);
        depth_vector.at(0) = brown_keypoints.at(3).x - brown_keypoints.at(0).x;
        depth_vector.at(1) = brown_keypoints.at(3).y - brown_keypoints.at(0).y;
        depth_vector.at(2) = brown_keypoints.at(3).z - brown_keypoints.at(0).z;

        norm = sqrt(pow(depth_vector.at(0), 2) + pow(depth_vector.at(1),2) + pow(depth_vector.at(2), 2));

        depth_vector.at(0) = depth_vector.at(0) / norm;
        depth_vector.at(1) = depth_vector.at(1) / norm;
        depth_vector.at(2) = depth_vector.at(2) / norm;

        std::vector<float> height_vector = std::vector<float> (3);
        height_vector.at(0) = unit_vector.at(0);
        height_vector.at(1) = unit_vector.at(1);
        height_vector.at(2) = unit_vector.at(2); // It must be same at any color


        double dist_array[width_step][depth_step][height_step] = {};
        int output_array[width_step][depth_step][height_step] = {};

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    dist_array[j][k][i] = 0;
                    output_array[j][k][i] = 0;
                }
            }
        }

        std::vector<int> index = std::vector<int> (3);

        for (int n = 0; n < brown_cloud->points.size(); n++)
        {                   
            pcl::PointXYZ pt;
            pt.x = brown_cloud->points[n].x;
            pt.y = brown_cloud->points[n].y;
            pt.z = brown_cloud->points[n].z;


            if(pt.x == 0 && pt.y == 0 && pt.z == 0)
            { }
            else{
                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            pcl::PointXYZ point;
                            point.x = brown_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            point.y = brown_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            point.z = brown_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            double dist = getDistant(pt, point);

                            dist_array[j][k][i] = dist;
                        }
                    }
                }

                double min_dist = 100;
                int min_i;
                int min_j;
                int min_k;

                for(int i = 0; i < height_step; i++)
                {
                    for(int j = 0; j < width_step; j++)
                    {
                        for(int k = 0; k < depth_step; k++)
                        {   
                            if(dist_array[j][k][i] <= min_dist)
                            { min_dist = dist_array[j][k][i]; min_i = i; min_j = j; min_k = k;} 
                        }
                    }
                }
                output_array[min_j][min_k][min_i] += 1;
            } 
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    cout << "value : " << output_array[j][k][i] << endl;
                }
            }
        }

        int max_value = 0;
        std::vector<int> max_index = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(output_array[j][k][i] > max_value)
                    {   
                        max_value = output_array[j][k][i];
                        max_index.at(0) = j;
                        max_index.at(1) = k;
                        max_index.at(2) = i;
                    }
                }
            }
        }

        int max_value_2 = 0;
        std::vector<int> max_index_2 = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                    {}
                    else
                    {
                        if(output_array[j][k][i] > max_value_2)
                        {   
                            max_value_2 = output_array[j][k][i];
                            max_index_2.at(0) = j;
                            max_index_2.at(1) = k;
                            max_index_2.at(2) = i;
                        }
                    }
                }
            }
        }

        int max_value_3 = 0;
        std::vector<int> max_index_3 = std::vector<int> (3);
        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if(j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2))
                    {}
                    else if(j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2))
                    {}
                    else{

                        if(output_array[j][k][i] > max_value_3)
                        {   
                            max_value_3 = output_array[j][k][i];
                            max_index_3.at(0) = j;
                            max_index_3.at(1) = k;
                            max_index_3.at(2) = i;
                        }
                    }
                }
            }
        }

        for(int i = 0; i < height_step; i++)
        {
            for(int j = 0; j < width_step; j++)
            {
                for(int k = 0; k < depth_step; k++)
                {   
                    if((j == max_index.at(0) && k == max_index.at(1) && i == max_index.at(2)) || (j == max_index_2.at(0) && k == max_index_2.at(1) && i == max_index_2.at(2)) || (j == max_index_3.at(0) && k == max_index_3.at(1) && i == max_index_3.at(2)))
                    {   
                        pcl::PointXYZ point;
                        point.x = brown_keypoints.at(0).x + height_vector.at(0)*(0.025)*i + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                        point.y = brown_keypoints.at(0).y + height_vector.at(1)*(0.025)*i + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                        point.z = brown_keypoints.at(0).z + height_vector.at(2)*(0.025)*i + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                        output_cloud.points.push_back(point);

                        if(i == 1)
                        {   
                            pcl::PointXYZ bottom_point;
                            bottom_point.x = brown_keypoints.at(0).x + height_vector.at(0)*(0.025)*(i - 1) + width_vector.at(0)*(0.025)*j + depth_vector.at(0)*(0.025)*k + (height_vector.at(0)*(0.025) + width_vector.at(0)*(0.025) + depth_vector.at(0)*(0.025)) / 2; 
                            bottom_point.y = brown_keypoints.at(0).y + height_vector.at(1)*(0.025)*(i - 1) + width_vector.at(1)*(0.025)*j + depth_vector.at(1)*(0.025)*k + (height_vector.at(1)*(0.025) + width_vector.at(1)*(0.025) + depth_vector.at(1)*(0.025)) / 2; 
                            bottom_point.z = brown_keypoints.at(0).z + height_vector.at(2)*(0.025)*(i - 1) + width_vector.at(2)*(0.025)*j + depth_vector.at(2)*(0.025)*k + (height_vector.at(2)*(0.025) + width_vector.at(2)*(0.025) + depth_vector.at(2)*(0.025)) / 2; 

                            output_cloud.points.push_back(bottom_point);
                        }
                    }
                }
            }
        }

        if(output_cloud.points.size() == 4)
            { pcl::io::savePLYFileASCII ("brown_output_cloud.ply", output_cloud); return output_cloud; }
    }      
}

void GetPose(std::string color_string, pcl::PointCloud<pcl::PointXYZ> color_synthetic)
{

}