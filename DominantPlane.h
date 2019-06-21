//
// Created by intelpro on 3/26/19.
//

//#ifndef DOMINANTPLANE_DOMINANTPLANE_H
//#define DOMINANTPLANE_DOMINANTPLANE_H
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <cstdlib>
#include <time.h>
using namespace std;
namespace Plane {
    struct Plane_model
    {
        float a;
        float b;
        float c;
        float d;
        float denominator;
        float avg_distance;
    };
    class DominantPlane {
    public:
        DominantPlane(float _fx, float _fy, float _cx, float _cy, float _DepthMapFactor, float _Distance_thresh,  int _max_iter, int _width, int _height);

        ~DominantPlane(void) {}

        cv::Mat Depth2pcd(cv::Mat &depth);
        Plane::Plane_model RunRansac(cv::Mat &pcd_inlier);
        void ResetValue()
        {
            N_pcds = 0;
            frame_id ++;
            pcd_concat.clear();
        };
        void FindPlaneFromSampledPcds(std::vector<cv::Vec3f> &sampled_pcds, Plane_model& Plane);
        void compute_inlier_cost(Plane_model& plane, cv::Mat& pcd_inlier, cv::Mat& PointCloud);
        void Object_Segmentation(cv::Mat& pcd_inlier, cv::Mat& pcd_object);

        void Object_Segmentation_2(Plane_model& plane, cv::Mat& pcd_object);

        static cv::Mat mK;
        static float DepthMapFactor;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float Distance_threshold;
        // number of point cloud
        static int max_iter;
        static int N_pcds;
        static int frame_id;
        static int N_inlier;
        static int width;
        static int height;
        static cv::Mat Pointcloud;
        std::vector<cv::Vec3f> pcd_concat;
    };
}
//#endif //DOMINANTPLANE_DOMINANTPLANE_H

