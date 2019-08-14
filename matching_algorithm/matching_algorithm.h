#include <iostream>
#include <vector>

using namespace std;

namespace Plane {
    struct Plane_model
    {
        float a; // normal vector x element
        float b; // normal vector y element
        float c; // normal vector z element
        float d; // plane constant ( ax+by+cz+d = 0 )
        float denominator; // used for calculating distance
        float avg_distance;
    };
    class DominantPlane {
    public:
        DominantPlane(float _fx, float _fy, float _cx, float _cy, float _DepthMapFactor, float _Distance_thresh,  int _max_iter, int _width, int _height);

        ~DominantPlane(void) {}

        cv::Mat Depth2pcd(cv::Mat &depth);
        Plane::Plane_model RunRansac(cv::Mat &pcd_inlier, cv::Mat point_cloud);
        void FindPlaneFromSampledPcds(std::vector<cv::Vec3f> &sampled_pcds, Plane_model& Plane);
        void compute_inlier_cost(Plane_model& plane, cv::Mat& pcd_inlier, cv::Mat& PointCloud);
        
        void ResetValue()
        {
            N_pcds = 0;
            frame_id ++;
            pcd_concat.clear();
        };
        
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
