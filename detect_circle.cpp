#include <librealsense2/rs.hpp>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <unistd.h>

#include "camera.h"

#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>

using namespace cv;
using namespace std;

pcl::PointCloud<pcl::PointXYZ>::Ptr PassThroughFilter (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
 double x_min , double x_max , double y_min, double y_max, double z_min, double z_max)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min, z_max);
    pass.filter(*cloud_filtered);
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(y_min, y_max);
    pass.filter(*cloud_filtered);
    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x_min, x_max);
    pass.filter(*cloud_filtered);

    return cloud_filtered;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr points_to_pcl(const rs2::points& points)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr GetPointCloud()
{
    rs2::pointcloud pc;
    rs2::points points;
    rs2::pipeline pipe;
    pipe.start();
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();
    points = pc.calculate(depth);

    auto pcl_points = points_to_pcl(points);

    return pcl_points;
}



int main(int argc, char* argv[])
{
    // Get the point cloud file.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    cloud = GetPointCloud();

    int cloud_size_origin = cloud->size();
    cout<<"The origin size of point cloud is "<<cloud_size_origin<<endl;



    pcl::visualization::PCLVisualizer viewer ("PointCloud look");

    int v1 (0);

    viewer.createViewPort (0.0, 0.0, 1.0, 1.0, v1);

    viewer.addCoordinateSystem(0.5);

    float bckgr_gray_level = 0.0;  // Black
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_in_color_h (cloud, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl, (int) 255 * txt_gray_lvl);
    viewer.addPointCloud (cloud, cloud_in_color_h, "cloud_in_v1", v1);

    viewer.addText ("The Original Point Cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
    // set color of the background
    viewer.setBackgroundColor (bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    // set camera pose
    viewer.setCameraPosition (0.1, 0.1, 1, 0, 0, 0, 0);
    // set window size
    viewer.setSize (1280, 1024);



//for depth camera
    
    int width = 640;
    int height = 480;

    rs2::pipeline p;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    rs2::align align_to(RS2_STREAM_COLOR);

    rs2::pipeline_profile selection = p.start(cfg);
    auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrin_ = depth_stream.get_intrinsics();
    intrinsic_param intrin;
    intrin.Set(intrin_.width, intrin_.height, intrin_.fx, intrin_.fy, intrin_.ppx, intrin_.ppy);

    float *pointcloud = new float[width*height*3];
    
    
    while(true)
    {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames(); 

        rs2::frame color_frame = frames.get_color_frame();
        cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
     
        // Get the depth frame's dimensions
        rs2::depth_frame depth = frames.get_depth_frame(); 
        cv::Mat depth_image = cv::Mat(height, width, CV_16UC1);
        cv::Mat depthColor = cv::Mat(depth_image.rows,depth_image.cols, CV_8UC3);
        depth_image.data = (unsigned char*)depth.get_data();


        GetPointCloud((unsigned short*)depth_image.data, width, height, intrin, pointcloud);

        convertDepthToColor(depth_image, depthColor);
     

        cv::namedWindow("color");
        cv::imshow("color", color_image);
        cvWaitKey(1);

        cv::namedWindow("depth2");
        cv::imshow("depth2", depthColor);
        cvWaitKey(1);


        //声明一个三通道图像，像素值全为0，用来将霍夫变换检测出的圆画在上面  
        Mat dst(color_image.size(), color_image.type());
        dst = Scalar::all(0);

        Mat src_gray;//彩色图像转化成灰度图  
        cvtColor(color_image, src_gray, CV_BGR2GRAY);
        threshold(src_gray, src_gray, 100, 255, CV_THRESH_OTSU);
        src_gray = 255 - src_gray;
        
        cv::namedWindow("gray");
        imshow("gray", src_gray);


        Mat bf;//对灰度图像进行双边滤波  
        int kvalue = 10;
        bilateralFilter(src_gray, bf, kvalue, kvalue * 2, kvalue / 2);
        //imshow("灰度双边滤波处理", bf);
        //imwrite("src_bf.png", bf);

        vector<Vec3f> circles;//声明一个向量，保存检测出的圆的圆心坐标和半径  
        HoughCircles(bf, circles, CV_HOUGH_GRADIENT, 1.5, 20, 130, 38, 1, 20);//霍夫变换检测圆  
        
        // show how many circles
        //cout << circles.size() << endl;
        
        
        for (size_t i = 0; i < circles.size(); i++)//把霍夫变换检测出的圆画出来  
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);

            //cvCircle(CvArr* img, CvPoint center, int radius, CvScalar color, int thickness=1, int lineType=8, int shift=0)
			//img为源图像指针
			//center为画圆的圆心坐标
			//radius为圆的半径
			//color为设定圆的颜色，规则根据B（蓝）G（绿）R（红）
			//thickness 如果是正数，表示组成圆的线条的粗细程度。否则，表示圆是否被填充
			//line_type 线条的类型。默认是8
			//shift 圆心坐标点和半径值的小数点位数
            circle(color_image, center, 0, Scalar(0, 255, 0), -1, 8, 0);
            circle(color_image, center, radius, Scalar(0, 0, 255), 2, 8, 0);

            //cout << cvRound(circles[i][0]) << "\t" << cvRound(circles[i][1]) << "\t"
                //<< cvRound(circles[i][2]) << endl;//在控制台输出圆心坐标和半径                
        }
        
        cv::namedWindow("circle");
        cv::imshow("circle", color_image);
        cvWaitKey(1);

    }
    
}

