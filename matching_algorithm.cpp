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


using namespace cv;
using namespace std;


cv::Mat DistantPlaneSegmentation(Plane::Plane_model plane, cv::Mat red_pcloud, float distance);
cv::Mat dominant_plane_projection(Plane::Plane_model plane, cv::Mat pCloud);
cv::Mat get_projected_image(Plane::Plane_model plane, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud);
void LetsFindBox(cv::Mat projected_image);
void FindBoundingBox(cv::Mat image_RGB, cv::Mat image_depth);
void BoundingBox(Plane::Plane_model dominant_plane, cv::Mat pCloud_outlier);
void CloudViewer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2);
pcl::PointXYZ GetUnitY(pcl::PointXYZ origin, pcl::PointXYZ unit_x, Plane::Plane_model plane);
pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_dominant_plane_projection(Plane::Plane_model plane, cv::Mat pCloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr back_projection(cv::Mat image);
pcl::PointCloud<pcl::PointXYZ>::Ptr makeForm (pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr cv2pcl(cv::Mat image);

cv::Scalar fitEllipseColor = Scalar(255,  0,  0);
cv::Mat processImage(cv::Mat image);
int sliderPos = 70;

cv::Mat image_RGB;
cv::Mat image_depth;

pcl::PointXYZ origin;
pcl::PointXYZ unit_x;
pcl::PointXYZ unit_y;

std::vector<pcl::PointXYZ> keypoints = std::vector<pcl::PointXYZ> (4);
std::vector<Point2f> rectpoints = std::vector<Point2f> (4);

float a;
float b;
float c;
float d;

bool minus_x;
bool minus_y; 

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
                Z = depth.at<uchar>(y,x) / DepthMapFactor;
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
    using namespace cv;

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

    for (int i = 0; i < RGB_image.rows; i++)
    {
        for (int j = 0; j < RGB_image.cols; j++)
        {
            if (red.at<uchar>(i, j) == 0) // ******choose color****** //
            {
                for (int k = 0; k < 3 ; k++)
                pCloud.at<cv::Vec3f>(i, j)[k] = 0;
            } 
        }
    
    }

}


void FindBoundingBox(cv::Mat image_RGB, cv::Mat image_depth)
{
    Plane::Plane_model dominant_plane;
    cv::Mat pCloud_outlier = cv::Mat::zeros(image_RGB.rows, image_RGB.cols, CV_32FC3);

    Segmentation(image_RGB, image_depth, dominant_plane, pCloud_outlier);

    Color_Segmentation(image_RGB, pCloud_outlier);

    BoundingBox(dominant_plane, pCloud_outlier);
}


int main()
{
    // Load depth image
    cv::Mat image_RGB;
    image_RGB = cv::imread("image_RGB.jpg", CV_LOAD_IMAGE_ANYCOLOR);

    cv::Mat image_depth;
    image_depth = cv::imread("image_depth.jpg", CV_LOAD_IMAGE_ANYDEPTH);
        
    Plane::Plane_model dominant_plane;
    cv::Mat pCloud_outlier = cv::Mat::zeros(image_RGB.rows, image_RGB.cols, CV_32FC3);
    Segmentation(image_RGB, image_depth, dominant_plane, pCloud_outlier);

    Color_Segmentation(image_RGB, pCloud_outlier);

    BoundingBox(dominant_plane, pCloud_outlier);

    return (0);
}


void BoundingBox(Plane::Plane_model dominant_plane, cv::Mat pCloud_outlier)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    outlier_cloud = cv2pcl(pCloud_outlier);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_projected_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl_projected_cloud = pcl_dominant_plane_projection(dominant_plane, pCloud_outlier);

    cv::Mat projected_image = get_projected_image(dominant_plane, pcl_projected_cloud);

    cv::Mat box_image = processImage(projected_image);

    pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    box_cloud = back_projection(box_image);

    pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    last_cloud = makeForm(box_cloud);

    CloudViewer(outlier_cloud, last_cloud);

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

    std::cerr << "Cloud after filtering: " << std::endl;
    std::cerr << *cloud_filtered << std::endl;
    

    double x_min = 10;
    double y_x;
    double z_x;

    double y_min = 10;
    double x_y;
    double z_y;

    double z_min = 10;
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


    origin.x = x_min;
    origin.y = y_min; // hyper parameter
    origin.z = (- x_min*plane.a - y_min*plane.b - plane.d) / plane.c; 

    pcl::PointXYZ x_dir;
    x_dir.x = x_min - origin.x;
    x_dir.y = y_x - origin.y;
    x_dir.z = z_x - origin.z;

    double norm = sqrt(pow(x_dir.x, 2) + pow(x_dir.y, 2) + pow(x_dir.z, 2));

    unit_x.x = x_dir.x / norm;
    unit_x.y = x_dir.y / norm;
    unit_x.z = x_dir.z / norm;

    cout << "origin: " << "( " << origin.x << ", " << origin.y << ", " << origin.z << " )" << endl;
    cout << "unit_x: " << "( " << unit_x.x << ", " << unit_x.y << ", " << unit_x.z << " )" << endl;
    cout << "plane: " << "( " << plane.a << ", " << plane.b << ", " << plane.c << ", " << plane.d << " )" << endl;

    
    cout << " put unit_y.x value" << endl;
    cin >> unit_y.x; // from matlab

    cout << " put unit_y.y value" << endl;
    cin >> unit_y.y; // from matlab

    cout << " put unit_y.z value" << endl;
    cin >> unit_y.z;// from matlab
    

    //unit_y = GetUnitY(origin, unit_x, plane);
    /*    
    unit_y.x = 0.9996;
    unit_y.y = 0.0185;
    unit_y.z = 0.0194;
    */
    

    //cout<< unit_x.x*unit_y.x + unit_x.y*unit_y.y + unit_x.z*unit_y.z << endl;
    int max_x = 0;
    int max_y = 0;

    for (int i = 0; i < cloud_filtered->points.size(); i++)
    {
        pcl::PointXYZ point = cloud_filtered->points[i];
        point.x -= origin.x;
        point.y -= origin.y;
        point.z -= origin.z;

        int x = (unit_x.x*point.x + unit_x.y*point.y + unit_x.z*point.z)*1000;
        int y = (unit_y.x*point.x + unit_y.y*point.y + unit_y.z*point.z)*1000;
        
        if(y < 0)
        {
            y = - y;
            minus_y = true;
        }

        if(x < 0)
        {
            x = -x;
            minus_x = true;
        }

        if(max_x < x) {max_x = x;};
        if(max_y < y) {max_y = y;};
    }


    cv::Mat projected_image = cv::Mat::zeros(max_x + 10, max_y + 10, CV_8UC1);

    for (int i = 0; i < cloud_filtered->points.size(); i++)
    {
        pcl::PointXYZ point = cloud_filtered->points[i];
        point.x -= origin.x;
        point.y -= origin.y;
        point.z -= origin.z;

        int x = (unit_x.x*point.x + unit_x.y*point.y + unit_x.z*point.z)*1000 + 5;
        int y = (unit_y.x*point.x + unit_y.y*point.y + unit_y.z*point.z)*1000 + 5;
        
        if(y < 0)
        {
            y = - y;
        }

        if(x < 0)
        {
            x = -x;
        }

        projected_image.at<uchar>(x, y) = 255;
    }

    cv::imwrite("projected_image.png", projected_image);

    a = plane.a;
    b = plane.b;
    c = plane.c;
    d = plane.d;

    return projected_image;
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

cv::Mat processImage(cv::Mat image)
{
 
    int thresh = 150;
    RNG rng(12345);

    Mat canny_output;
    Canny( image, canny_output, thresh, thresh*2, 3 );
    vector<vector<Point> > contours;
    findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
        if( contours[i].size() > 5 )
        {
            minEllipse[i] = fitEllipse( contours[i] );
        }
    }
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    
    double max_size = 300;

    for( size_t i = 0; i< contours.size(); i++ )
    {
        // contour
        //drawContours( drawing, contours, (int)i, color ); // we don't need contour!
        // rotated rectangle
        Point2f rect_points[4];
        minRect[i].points( rect_points );
        
        double dist_1 = sqrt(pow(rect_points[0].x - rect_points[1].x, 2) + pow(rect_points[0].y - rect_points[1].y, 2));
        double dist_2 = sqrt(pow(rect_points[1].x - rect_points[2].x, 2) + pow(rect_points[1].y - rect_points[2].y, 2));

        cout << "Length of rectangle: " << dist_1 << ", " << dist_2 << endl;

        double size = dist_1 * dist_2;
        if(max_size < size)
        {
            max_size = size;
            for ( int j = 0; j < 4; j++ )
            {   
                rectpoints.at(j) = rect_points[j];
            }  
        }

    }

    Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );

    for ( int j = 0; j < 4; j++ )
    {   
        line( drawing, rectpoints.at(j), rectpoints.at((j+1)%4), color );
    } 


    imshow(" drawing ", drawing);
    //waitKey(0);

    return drawing;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr back_projection(cv::Mat image)
{       
    pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if(image.at<cv::Vec3b>(i, j)[0] == 0 && image.at<cv::Vec3b>(i, j)[1] == 0 && image.at<cv::Vec3b>(i, j)[2] == 0)
            {}
            else 
            {   

                double i_ = i - 5;
                double j_ = j - 5;

                i_ /= 1000;
                j_ /= 1000;

                if (minus_x == true)
                {
                    i_ = - i_;
                }

                if (minus_y == true)
                {
                    j_ = - j_;
                }

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
        double i_ = rectpoints.at(i).y - 5;
        double j_ = rectpoints.at(i).x - 5;

        i_ /= 1000;
        j_ /= 1000;

        if (minus_x == true)
        {
            i_ = - i_;
        }


        if (minus_y == true)
        {
            j_ = - j_;
        }


        double x = i_*(unit_x.x) + j_*(unit_y.x) + origin.x;
        double y = i_*(unit_x.y) + j_*(unit_y.y) + origin.y;
        double z = i_*(unit_x.z) + j_*(unit_y.z) + origin.z;

        keypoints.at(i).x = x;
        keypoints.at(i).y = y;
        keypoints.at(i).z = z;
    }

    return box_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr makeForm (pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp (new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<float> unit_vector = std::vector<float> (3);
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

        double scalar = 0.05;

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

        double dist = 0.05;
        double intval = dist / 20;

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
    pcl::PointXYZ best_unit_y;

    double best_total_cost = 100;
    double total_cost;
    double cost_1;
    double cost_2;
    double cost_3;

    int max_iter = 100000;

    double thresh = 0.0005;


    for (int i = 0; i < max_iter; i++)
    {   
        unit_y.x = std::rand();
        unit_y.y = std::rand();
        unit_y.z = std::rand();

        cost_1 = unit_x.x*unit_y.x + unit_x.y*unit_y.y + unit_x.z*unit_y.z;
        cost_2 = (unit_y.x + origin.x)*plane.a + (unit_y.y + origin.y)*plane.b + (unit_y.z + origin.z)*plane.c + plane.d; 
        cost_3 = unit_y.x*unit_y.x + unit_y.y*unit_y.y + unit_y.z*unit_y.z - 1;

        total_cost = cost_1 + cost_2 + cost_3;

        if(total_cost < thresh)
        {   
            cout << "unit_y: " << "( " << unit_y.x << ", " << unit_y.y << ", " << unit_y.z << " )" << endl;
            return unit_y;
        }

        if(total_cost < best_total_cost)
        {
            best_total_cost = total_cost;
            best_unit_y = unit_y;
        }
    }

    cout << "unit_y: " << "( " << best_unit_y.x << ", " << best_unit_y.y << ", " << best_unit_y.z << " )" << endl;
    return best_unit_y;
}