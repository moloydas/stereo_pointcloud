#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <ros/ros.h>
#include <camera_calibration_parsers/parse_ini.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/ximgproc.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

ros::Publisher pc_pub;
cv_bridge::CvImagePtr left_cv_ptr;
cv_bridge::CvImagePtr right_cv_ptr;

cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm;

clock_t start, end;


int minDisparity = 0;
int blockSize = 5;
int numDisparities = 320;
int preFilterCap = 31;
int uniquenessRatio = 10;
int speckleWindowSize = 150;
int speckleRange = 8;
int disp12MaxDiff = 10;
int P1 = 8*3*blockSize*blockSize;
int P2 = 32*3*blockSize*blockSize;

void image_callback(const sensor_msgs::ImageConstPtr& left_img_msg, const sensor_msgs::ImageConstPtr& right_img_msg){
    try{
      left_cv_ptr = cv_bridge::toCvCopy(left_img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    try{
      right_cv_ptr = cv_bridge::toCvCopy(right_img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat disp,disp8,raw_dis;
    cv::Mat left_img_color,right_img_color;
    cv::Mat left_img,right_img;
    cv::resize(left_cv_ptr->image ,left_img_color ,cv::Size(),0.5,0.5);
    cv::resize(right_cv_ptr->image ,right_img_color ,cv::Size(),0.5,0.5);
    cv::cvtColor(left_img_color,  left_img,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_img_color, right_img, cv::COLOR_BGR2GRAY);

    start = clock();
    sgbm->compute( left_img, right_img, disp);
    cv::ximgproc::getDisparityVis(disp, disp8, 1);
    // cv::normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    end = clock();

    std::cout << "disp size:" << disp.size() << " compute time: " << double(end-start)/double(CLOCKS_PER_SEC)<< std::endl;

    cv::Mat stereo_image(left_cv_ptr->image.rows, (left_cv_ptr->image.cols)*2, CV_8UC3, cv::Scalar(0,0,0));
    left_cv_ptr->image.copyTo(stereo_image(cv::Rect(0,0,left_cv_ptr->image.cols,left_cv_ptr->image.rows)));
    right_cv_ptr->image.copyTo(stereo_image(cv::Rect(left_cv_ptr->image.cols,0,right_cv_ptr->image.cols,right_cv_ptr->image.rows)));
    // cv::imshow("stereo_image", stereo_image);
    cv::imshow("stereo_image", left_img_color);
    cv::imshow("disparity", disp8);
    cv::waitKey(30);
}

int main(int argc, char **argv){

    ros::init(argc, argv, "Pointcloud_Generation_using_Stereo_Vision");

    ros::NodeHandle n;
    image_transport::ImageTransport it(n);

    message_filters::Subscriber<sensor_msgs::Image> image_left_sub(n, "zed/zed_node/left/image_rect_color", 1);
    message_filters::Subscriber<sensor_msgs::Image> image_right_sub(n, "zed/zed_node/right/image_rect_color", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_left_sub, image_right_sub);
    sync.registerCallback(boost::bind(&image_callback, _1, _2));

    pc_pub = n.advertise<sensor_msgs::PointCloud2>("pointcloud", 1);

    left_cv_ptr = cv_bridge::CvImagePtr(new cv_bridge::CvImage);
    right_cv_ptr = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    sgbm = cv::stereo::StereoBinarySGBM::create(minDisparity,
                                                numDisparities,
                                                blockSize,
                                                P1,
                                                P2,
                                                disp12MaxDiff,
                                                preFilterCap,
                                                uniquenessRatio,
                                                speckleWindowSize,
                                                speckleRange,
                                                cv::StereoSGBM::MODE_SGBM );

    cv::namedWindow("stereo_image",cv::WINDOW_NORMAL);	
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::startWindowThread();
    ros::Rate loop_rate(20);

    ROS_INFO("\nStereo PointCloud node started!!!\n");

    while(ros::ok()){

        ros::spinOnce();

        loop_rate.sleep();
    }

    cv::destroyWindow("view");
    return 0;
}
