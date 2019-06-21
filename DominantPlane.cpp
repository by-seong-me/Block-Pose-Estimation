#include "DominantPlane.h"

//
// Created by intelpro on 3/26/19.
//

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

    Plane_model DominantPlane::RunRansac(cv::Mat &pcd_inlier) {
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
            compute_inlier_cost(plane, pcd_temp_inlier, Pointcloud);
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

        /*
        cv::Mat pcd_outlier = cv::Mat::zeros(height, width, CV_32FC3);
        for (int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                pcd_outlier.at<cv::Vec3f>(y, x) = Pointcloud.at<cv::Vec3f>(y, x) - pcd_inlier.at<cv::Vec3f>(y, x);
            }
        }

        cv::Mat pcd_inlier_2 = cv::Mat::zeros(height, width, CV_32FC3);
        Plane_model plane_2;
        //best_N_inlier = 0;
        float best_cost_2 = std::numeric_limits<float>::infinity();
        for (int i = 0; i < max_iter/2; i++)
        {
            cv::Mat pcd_temp_inlier = cv::Mat::zeros(height, width, CV_32FC3);
            std::vector<cv::Vec3f> sampled_pcd2;
            for(int j =0; j < 3; j++)
            {
                rand_idx = int(pcd_concat.size() * (std::rand()/(double)RAND_MAX));
                sampled_pcd2.push_back(pcd_concat.at(rand_idx));
            }
            FindPlaneFromSampledPcds(sampled_pcd2, plane_2);
            compute_inlier_cost(plane_2, pcd_temp_inlier, pcd_outlier);
            cost = plane_2.avg_distance / N_inlier;
            if(best_cost_2 > cost)
            {
                best_cost_2 = cost;
                pcd_inlier_2 = pcd_temp_inlier;
                cout << "inlier2/" << "iter:" << i << " " << "cost:" << cost << " " << "average distance:" << plane.avg_distance << " " << endl;
            }

            N_inlier = 0;
            pcd_temp_inlier.release();
            sampled_pcd2.clear();
        }

        for (int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                pcd_inlier.at<cv::Vec3f>(y, x) += pcd_inlier_2.at<cv::Vec3f>(y, x);
            }
        }

        */

        ResetValue();
        return best_plane;
    }

    void DominantPlane::Object_Segmentation(cv::Mat& pcd_inlier, cv::Mat& pcd_object)
    {
        float Threshold = 1.5;
        float differ = 0;
        float differ_2 = 0;
        for (int i=0; i<pcd_inlier.rows; i++)
        {
            for(int j=0; j<pcd_inlier.cols; j++)
            {
                float sum = 0;
                float sum_2 = 0;
                for (int k=0; k<3; k++)

                {
                    differ = Pointcloud.at<cv::Vec3f>(i, j)[k] - pcd_inlier.at<cv::Vec3f>(i, j)[k];
                    differ_2 = differ * differ;
                    sum += differ_2;

                    sum_2 += pcd_inlier.at<cv::Vec3f>(i, j)[k] * pcd_inlier.at<cv::Vec3f>(i, j)[k];
                }
                if (sum_2 == 0)
                {
                    if (sum < Threshold)
                    {
                        pcd_object.at<cv::Vec3f>(i, j) = Pointcloud.at<cv::Vec3f>(i, j);

                    }

                }
            }
        }

    }

    void DominantPlane::Object_Segmentation_2(Plane_model& plane, cv::Mat& pcd_object)
    {
        float Threshold = 0.2;
        for (int i=0; i<pcd_object.rows; i++)
        {
            for(int j=0; j<pcd_object.cols; j++)
            {
                float x = Pointcloud.at<cv::Vec3f>(i, j)[0];
                float y = Pointcloud.at<cv::Vec3f>(i, j)[1];
                float z = Pointcloud.at<cv::Vec3f>(i, j)[2];
                float dist = abs(plane.a * x + plane.b * y + plane.c * z + plane.d) / plane.denominator;

                if (0.01 < dist && dist < Threshold)
                {
                    pcd_object.at<cv::Vec3f>(i, j) = Pointcloud.at<cv::Vec3f>(i, j);

                }
            }

        }


    }

    void DominantPlane::FindPlaneFromSampledPcds(std::vector<cv::Vec3f> &sampled_pcds, Plane_model& plane)
    {
        cv::Vec3f A = sampled_pcds[0], B = sampled_pcds[1], C = sampled_pcds[2];
        // cout << "A" << A << "B" << B << endl;
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
    /*
    def compute_inlier_cost(plane_model, point_cloud, threshold = 0.001):
    distance = plane_model.get_distance_from_pcd(point_cloud)
    inlier_pcd = point_cloud[distance < threshold, :]
    return inlier_pcd, distance
    */
}
























