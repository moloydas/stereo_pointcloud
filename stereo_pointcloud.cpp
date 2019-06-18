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

cv::Ptr<cv::StereoSGBM> left_sgbm;
cv::Ptr<cv::StereoMatcher> right_sgbm;
cv::Ptr< cv::ximgproc::DisparityWLSFilter> wls_filter;

clock_t start, end;

typedef pcl::PointXYZRGB  PointType;
pcl::PointCloud<PointType>::Ptr pc;
PointType pt;

float Q_matrix[] = {1,0,0,-606.146,0,1,0,-332.562,0,0,0,699.921,0,0,1/.120193,0};
cv::Mat Q_mat(4,4,CV_32FC1,Q_matrix);
cv::Mat _3d_img;

int minDisparity = 0;
int blockSize = 3;
int numDisparities = 32;
int preFilterCap = 31;
int uniquenessRatio = 10;
int speckleWindowSize = 150;
int speckleRange = 8;
int disp12MaxDiff = 10;
int P1 = 8*3*blockSize*blockSize;
int P2 = 32*3*blockSize*blockSize;
int lambda = 10;
int sigma = 25;

void set_params(){
    left_sgbm->setMinDisparity(minDisparity);
    if (preFilterCap % 2 == 0){
        preFilterCap = preFilterCap + 1;
    }
    left_sgbm->setPreFilterCap(preFilterCap);
    left_sgbm->setUniquenessRatio(uniquenessRatio);
    left_sgbm->setSpeckleWindowSize(speckleWindowSize);
    left_sgbm->setSpeckleRange(speckleRange);
    left_sgbm->setDisp12MaxDiff(disp12MaxDiff);
    if (P1 >= P2/2){
        P2 = P1*2 + 1;
        cv::setTrackbarPos("P2","Track Bar Window",P2);
    }
    else if(P1 <= 0){
        P1 = 1;
        cv::setTrackbarPos("P1","Track Bar Window",P1);
    }
    else if(P2 <= 0){
        P2 = 1;
        cv::setTrackbarPos("P2","Track Bar Window",P2);
    }
    left_sgbm->setP2(P2);
    left_sgbm->setP1(P1);
}

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
    cv::Mat left_disp,right_disp,filtered_disp;
    cv::Mat left_img_color,right_img_color;
    cv::Mat left_img,right_img;
    cv::resize(left_cv_ptr->image ,left_img_color ,cv::Size(),0.5,0.5);
    cv::resize(right_cv_ptr->image ,right_img_color ,cv::Size(),0.5,0.5);
    cv::cvtColor(left_img_color,  left_img,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_img_color, right_img, cv::COLOR_BGR2GRAY);

    start = clock();
    left_sgbm->compute( left_img, right_img, left_disp);
    right_sgbm->compute( right_img, left_img, right_disp);
    wls_filter->setLambda( ((float)lambda) * 0.1);
    wls_filter->setSigmaColor(sigma);
    wls_filter->filter(left_disp,left_cv_ptr->image,filtered_disp,right_disp);
    cv::ximgproc::getDisparityVis(filtered_disp, disp8, 1);
    // cv::normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    end = clock();

    double BaseLine = 120.193;
    double f = 699.921;
    uint32_t r,g,b,rgb;
    double disparity = 0;

    pc->clear();
    cv::reprojectImageTo3D(disp8, _3d_img, Q_mat, true, CV_32F);

    cv::Vec3b pixel;
    for(int i=0; i<filtered_disp.rows; i++){
        for(int j=0; j<filtered_disp.cols; j++){
			disparity = filtered_disp.at<double>(i,j);
			if (disparity == 0) continue;
			cv::Point3f p = _3d_img.at<cv::Point3f>(i, j);
			pt.x = p.x;
			pt.y = p.y;
			pt.z = p.z;
            pixel = left_cv_ptr->image.at<cv::Vec3b>(i,j);
            b = pixel[0];
            g = pixel[1];
            r = pixel[2];
            rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
            pt.rgb = *reinterpret_cast<float*>(&rgb);
            pc->push_back(pt);
        }
    }

    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg(*pc, pc_msg);
    pc_msg.header.frame_id = "left_camera";
    pc_msg.header.stamp = ros::Time::now();
    pc_pub.publish(pc_msg);

    std::cout << "disp size:" <<filtered_disp.size() << " compute time: " << double(end-start)/double(CLOCKS_PER_SEC)<< std::endl;

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

    left_sgbm = cv::StereoSGBM::create(minDisparity,
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

    wls_filter = cv::ximgproc::createDisparityWLSFilter(left_sgbm);
    right_sgbm = cv::ximgproc::createRightMatcher(left_sgbm);

    cv::namedWindow("stereo_image",cv::WINDOW_NORMAL);	
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::namedWindow("Track Bar Window", CV_WINDOW_NORMAL);
    cvCreateTrackbar("Pre Filter Cap", "Track Bar Window", &preFilterCap, 61);
    cvCreateTrackbar("Minimum Disparity", "Track Bar Window", &minDisparity, 200);
    cvCreateTrackbar("Uniqueness Ratio", "Track Bar Window", &uniquenessRatio, 30);
    cvCreateTrackbar("Speckle Range", "Track Bar Window", &speckleRange, 500);
    cvCreateTrackbar("Speckle Window Size", "Track Bar Window", &speckleWindowSize, 300);
    cvCreateTrackbar("P1", "Track Bar Window", &P1, 10000);
    cvCreateTrackbar("P2", "Track Bar Window", &P2, 10000);
    cvCreateTrackbar("lambda", "Track Bar Window", &lambda, 100);
    cvCreateTrackbar("sigma", "Track Bar Window", &sigma, 255);

    cv::startWindowThread();

    pc = pcl::PointCloud<PointType>::PointCloud::Ptr(new pcl::PointCloud<PointType>);

	std::cout << Q_mat << std::endl;

    ros::Rate loop_rate(20);

    ROS_INFO("\nStereo PointCloud node started!!!\n");

    while(ros::ok()){

        // set_params();
        ros::spinOnce();

        loop_rate.sleep();
    }

    cv::destroyWindow("view");
    return 0;
}
