
# include <iostream>
# include <fstream>
# include <string>
# include <opencv2/opencv.hpp>
# include <opencv2/viz.hpp>
# include <time.h>
# include <ros/ros.h>

# include "DominantPlane.h"

# include "sensor_msgs/image_encodings.h"
# include "cv_bridge/cv_bridge.h"
# include <image_transport/image_transport.h>
# include <opencv2/imgproc/imgproc.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <message_filters/subscriber.h>
# include <message_filters/time_synchronizer.h>

#include <ctime>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <pcl/search/impl/search.hpp>

#ifndef PCL_NO_PRECOMPILE
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_types.h>
PCL_INSTANTIATE(Search, PCL_POINT_TYPES)
#endif // PCL_NO_PRECOMPILE

int number = 0;

using namespace std;
using namespace cv;
cv::Mat imRGB, imDepth;
// pointCloud;
namespace enc = sensor_msgs::image_encodings;

//typedef boost::shared_ptr< cv_bridge::CvImage > CvImagePtr;

void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
void DepthCallback(const sensor_msgs::ImageConstPtr& msg);
//void pCloudCallback(const sensor_msgs::ImageConstPtr& msg);

void Get_RGB(cv_bridge::CvImagePtr& cv_ptr);
void Get_Depth(cv_bridge::CvImagePtr& cv_ptr);

void Segmentation(cv::Mat& image_RGB, cv::Mat& image_Depth);

void Show_Results(cv::Mat& pointCloud, cv::Mat RGB_image_original);

cv::Mat imageCb(cv::Mat& RGB_image);
cv::Mat RGB2pcl(cv::Mat red_image, cv::Mat pointCloud);

void registration(cv::Mat& pointCloud_output);

pcl::PointCloud<pcl::PointXYZ>::Ptr MatToPoinXYZ(cv::Mat OpencVPointCloud);

int main(int argc, char** argv)
{
    ros::init(argc, argv,"segmentation");
    ros::NodeHandle nh;

    //Mat imRGB, imDepth;
    ros::Subscriber sub_color = nh.subscribe("/camera/color/image_raw", 100, ImageCallback);
    ros::Subscriber sub_depth = nh.subscribe("/camera/aligned_depth_to_color/image_raw", 100, DepthCallback);
    //ros::Subscriber sub_pCloud = nh.subscribe("/camera/depth/color/points", 100, pCloudCallback);



    ros::spin();

    return 0;

}


void ImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    ROS_INFO("Color image height = %d, width = %d", msg->height, msg->width);

    cv_bridge::CvImagePtr cv_ptrRGB;

    try
    {
        cv_ptrRGB = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        Get_RGB(cv_ptrRGB);

    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}

void DepthCallback(const sensor_msgs::ImageConstPtr& msg)
{
    number++;
    ROS_INFO("Depth image height = %d, width = %d", msg->height, msg->width);

    cv_bridge::CvImagePtr cv_ptrD;
    try {
        cv_ptrD = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        Get_Depth(cv_ptrD);
        if (imRGB.empty() || imDepth.empty()) {
            cerr << endl << "Failed to load image at: " << endl;
            return;
        } else if(number%6 == 0){

            Segmentation(imRGB, imDepth);
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}

/*
void pCloudCallback(const sensor_msgs::ImageConstPtr& msg)
{
    ROS_INFO("Cloud image height = %d, width = %d", msg->height, msg->width);

    cv_bridge::CvImagePtr cv_ptrCloud;
    try {
        cv_ptrCloud = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        Get_pCloud(cv_ptrCloud);
        if (pointCloud.empty()) {
            cerr << endl << "Failed to load image at: " << endl;
            return;
        }
        else {
            Segmentation(imRGB, imDepth, pointCloud);
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}
 */

void Get_RGB (cv_bridge::CvImagePtr& cv_ptr)
{
    imRGB = cv_ptr->image;
    cout << "row of Color image" << imRGB.rows << endl;
}

void Get_Depth (cv_bridge::CvImagePtr& cv_ptr)
{
    imDepth = cv_ptr->image;
    cout << "row of Depth image" << imDepth.rows << endl;
}

/*
void Get_pCloud (cv_bridge::CvImagePtr& cv_ptr)
{
    pointCloud = cv_ptr->image;
    cout << "row of Cloud image" << pointCloud.rows << endl;
}
*/

void Segmentation(cv::Mat& image_RGB, cv::Mat& image_Depth)
{
    cout << "I'm stucked" << endl;

    float fx = 615.6707153320312;
    //float fx = 520.9;
    float fy = 615.962158203125;
    //float fy = 521.0;
    float cx = 328.0010681152344;
    //float cx = 325.1;
    float cy = 241.31031799316406;
    //float cy = 249.7;
    float scale = 1000;
    float Distance_theshold = 0.005;
    int width = 640;
    int height = 480;
    int max_iter = 300;
    Plane::DominantPlane plane(fx,fy,cx,cy, scale, Distance_theshold, max_iter, width, height);

    clock_t start, end;

    cv::Mat pCloud(height, width, CV_32FC3);
    cv::Mat pCloud_inlier(height, width, CV_32FC3);

    start = clock();
    cv::Mat pointCloud = plane.Depth2pcd(imDepth);

    cv::Mat pcd_outlier = cv::Mat::zeros(height, width, CV_32FC3);
    Plane::Plane_model best_plane = plane.RunRansac(pCloud_inlier);

    cv::Mat pCloud_outlier = cv::Mat::zeros(height, width, CV_32FC3);

    for (int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            pCloud_outlier.at<cv::Vec3f>(y, x) = pointCloud.at<cv::Vec3f>(y, x) - pCloud_inlier.at<cv::Vec3f>(y, x);

        }
    }

    cv::Mat pcd_object = cv::Mat::zeros(height, width, CV_32FC3);

    //plane.Object_Segmentation(pCloud_inlier, pcd_object);
    plane.Object_Segmentation_2(best_plane, pcd_object);

    end = clock();
    double result = (double)(end - start)/CLOCKS_PER_SEC;

    cout << result << endl;
    /*
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);

    imwrite("pc_" + str, pointCloud);
    imwrite("pc_in_" + str, pCloud_inlier);
    imwrite("pc_ob_" + str, pcd_object);
    */


    // viz::Viz3d myWindow("Point Cloud");
    // viz::Viz3d myWindow_2("Point Cloud 2");
    // viz::Viz3d myWindow_3("Point Cloud 3");//rty

    //cv::Mat al_imRGB = AlignRGB(imRGB);

    // viz::WCloud wcloud(pointCloud, imRGB); // 0
    // viz::WCloud wcloud_2(pCloud_inlier, imRGB);
    // viz::WCloud wcloud_3(pcd_object, imRGB);//try

    //viz::WCloud wcloud(pCloud_inlier, imRGB); // 1
    //viz::WCloud wcloud(pcd_object, imRGB); // 2
    //viz::WCloud wcloud(pCloud_inlier);

    // myWindow.showWidget("CLOUD", wcloud);
    // myWindow_2.showWidget("CLOUD_2", wcloud_2);
    // myWindow_3.showWidget("CLOUD_3", wcloud_3);//try

    //myWindow.spin();
    //myWindow_2.spinOnce(2, true);


    //namedWindow( "RGB_image", WINDOW_AUTOSIZE );
    //imshow("RGB_image", imRGB);

    // myWindow.spin();
    // myWindow_2.spin();
    // myWindow_3.spin();



    Show_Results(pcd_object, imRGB);
    //*Show_Results_2(pCloud_inlier, imRGB);
    //myWindow.spinOnce(2, true);

    // voxel_grid(pCloud_inlier); //show voxel


}

void Show_Results(cv::Mat& pointCloud, cv::Mat RGB_image_original)
{
    cv::Mat RGB_image = RGB_image_original.clone();

    for (int y = 0; y < RGB_image.rows; y++)
    {
        for(int x = 0; x < RGB_image.cols; x++) {

            if (pointCloud.at<cv::Vec3f>(y, x)[0] == 0 && pointCloud.at<cv::Vec3f>(y, x)[1] == 0 && pointCloud.at<cv::Vec3f>(y, x)[2] == 0) {
                //cout << " I'm here~ " << endl;
                RGB_image.at<Vec3b>(y, x)[0] = 0;
                //cout << RGB_image.at<Vec3b>(y, x)[0] << RGB_image.at<Vec3b>(y, x)[1] << endl;
                RGB_image.at<Vec3b>(y, x)[1] = 0;
                RGB_image.at<Vec3b>(y, x)[2] = 0;
            }
        }

    }


    cv::Mat red_image;
    red_image = imageCb(RGB_image);

    // namedWindow( "Point_cloud", WINDOW_NORMAL);
    // imshow("Point_cloud", pCloud_inlier);

    namedWindow( "RGB_image_seg", WINDOW_NORMAL );
    imshow("RGB_image_seg", RGB_image);

    waitKey(2);

    cv::Mat pointCloud_output;
    pointCloud_output = RGB2pcl(red_image, pointCloud);

    //registration(pointCloud_output);

}

/*
void Show_Results_2(cv::Mat RGB_seg_image, cv::Mat RGB_image_original)
{
    cv::Mat RGB_image = RGB_image_original.clone();

    for (int y = 0; y < RGB_image.rows; y++)
    {
        for(int x = 0; x < RGB_image.cols; x++) {

            if (RGB_seg_image.at<Vec3b>(y, x)[0] == 0 && RGB_seg_image.at<Vec3b>(y, x)[1] == 0 && RGB_seg_image.at<Vec3b>(y, x)[2] == 0) {
            }
            else{
                RGB_image.at<Vec3b>(y, x)[0] = 0;
                //cout << RGB_image.at<Vec3b>(y, x)[0] << RGB_image.at<Vec3b>(y, x)[1] << endl;
                RGB_image.at<Vec3b>(y, x)[1] = 0;
                RGB_image.at<Vec3b>(y, x)[2] = 0;
            }
        }

    }

    namedWindow( "RGB_image_2", WINDOW_AUTOSIZE );
    imshow("RGB_image_2", RGB_image);
    waitKey(2);

}
*/

cv::Mat imageCb(cv::Mat& RGB_image)
{
    static const std::string OPENCV_WINDOW2 = "Image window2";

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


    cv::imshow(OPENCV_WINDOW2, red); // change the color
    cv::waitKeyEx(2);

    return red;
}


cv::Mat RGB2pcl(cv::Mat red_image, cv::Mat pointCloud)
{
    static const std::string OPENCV_WINDOW = "Image window";

    cv::Mat pointCloud_output = pointCloud.clone();

    for (int y = 0; y < red_image.rows; y++)
    {
        for(int x = 0; x < red_image.cols; x++) {

            if (red_image.at<uchar>(y, x) == 0) {
                pointCloud_output.at<cv::Vec3f>(y, x)[0] = 0;
                pointCloud_output.at<cv::Vec3f>(y, x)[1] = 0;
                pointCloud_output.at<cv::Vec3f>(y, x)[2] = 0;
            }
        }

    }

    cv::imshow(OPENCV_WINDOW, pointCloud_output); // change the color
    cv::waitKeyEx(2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    cloud = MatToPoinXYZ(pointCloud_output);

    pcl::io::savePLYFileASCII ("test_pcd.ply", *cloud);

    return pointCloud_output;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr MatToPoinXYZ(cv::Mat OpencVPointCloud)
         {
             /*
             *  Function: Get from a Mat to pcl pointcloud datatype
             *  In: cv::Mat
             *  Out: pcl::PointCloud
             */

             //char pr=100, pg=100, pb=100;
             pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);//(new pcl::pointcloud<pcl::pointXYZ>);

             for(int i=0;i<OpencVPointCloud.cols;i++)
             {
                 for(int j=0; j<OpencVPointCloud.rows; j++)
                 {
                     pcl::PointXYZ point;
                     point.x = OpencVPointCloud.at<cv::Vec3f>(j,i)[0];
                     point.y = OpencVPointCloud.at<cv::Vec3f>(j,i)[1];
                     point.z = OpencVPointCloud.at<cv::Vec3f>(j,i)[2];

                     point_cloud_ptr -> points.push_back(point);

                 }
                //std::cout<<i<<endl;


                // when color needs to be added:
                //uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
                //point.rgb = *reinterpret_cast<float*>(&rgb);

             }
             // point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
             // point_cloud_ptr->height = 1;

             return point_cloud_ptr;

         }




void registration(cv::Mat& pointCloud_output)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_in = MatToPoinXYZ(pointCloud_output);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile<pcl::PointXYZ> ("test_ply.ply", *cloud_out);

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp; // RGB?

    icp.setInputSource(cloud_in);
    icp.setInputTarget(cloud_out);

    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

}



