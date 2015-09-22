#include <iostream>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "sample_config.hpp"
#include "features_detect.hpp"

void drawObject(cv::Mat img, std::vector<cv::Point2f> &pt, cv::Point2f offset_point)
{
	static cv::Scalar color(255, 255, 0);
	static int thickness = 3;
	static int lineType = cv::LINE_AA;
	static int shift = 0;
	static double t = 0;
	if(t >= 6.28)
		t = 0;
	else
		t = t + 0.04;
	//边框
	//cv::line(img, pt[0] + offset_point, pt[1] + offset_point, color, thickness, lineType, shift);
	//cv::line(img, pt[1] + offset_point, pt[2] + offset_point, color, thickness, lineType, shift);
	//cv::line(img, pt[2] + offset_point, pt[3] + offset_point, color, thickness, lineType, shift);
	//cv::line(img, pt[3] + offset_point, pt[0] + offset_point, color, thickness, lineType, shift);

	//单点
	cv::circle(img, pt[4]+offset_point, 80, color, 2, lineType);
	//巡逻点
	cv::circle(img, cv::Point2f(pt[4].x+std::cos(t)*80, pt[4].y+std::sin(t)*80)+offset_point,
			8, color, cv::FILLED, lineType);
	//物体名字
	cv::putText(img, OBJECT_IMG, cv::Point2f(pt[4].x-80, pt[4].y-100)+offset_point,
			cv::FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType);
}

void hessianTrackbarCallback(int thres, void* in_featuresDetect)
{
	CfeaturesDetect* featuresDetect =  (CfeaturesDetect*) in_featuresDetect;
	featuresDetect->m_surf->setHessianThreshold(thres);
	featuresDetect->init();
}

void minValueTrackbarCallback(int thres, void* in_featuresDetect)
{
	CfeaturesDetect* featuresDetect =  (CfeaturesDetect*) in_featuresDetect;
	featuresDetect->m_goodMatchMinValue = thres / 100.0;
}

int main(void)
{
	cv::Mat object_image = cv::imread(OBJECT_IMG);
	cv::Mat scene_image, match_image;

	cv::VideoCapture camera(CAMERA_NUM);
	camera.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_FRAME_HEIGHT);
	camera.set(cv::CAP_PROP_FRAME_WIDTH, CAM_FRAME_WIDTH);
	cv::namedWindow("Features Detect");

	CfeaturesDetect featuresDetect;
	featuresDetect.m_objectImage = object_image;
	featuresDetect.init();

	std::vector<cv::Point2f> object_corners(5), scene_corners(5);
	object_corners[0] = cv::Point2f(0,0);
	object_corners[1] = cv::Point2f(object_image.cols, 0);
	object_corners[2] = cv::Point2f(object_image.cols, object_image.rows);
	object_corners[3] = cv::Point2f(0, object_image.rows);
	object_corners[4] = cv::Point2f(object_image.cols/2, object_image.rows/2);

	cv::namedWindow("Config");
	int initMinValue = featuresDetect.m_goodMatchMinValue * 100.0;
	cv::createTrackbar("Hessian Thresold", "Config", &featuresDetect.m_hessianThresold,
						10000, hessianTrackbarCallback, (void*)&featuresDetect);
	cv::createTrackbar("Good Match Distance Times", "Config", &featuresDetect.m_goodMatchDistanceTimes,
						10, NULL);
	cv::createTrackbar("Good Match Min Value(100 times)", "Config", &initMinValue,
						50, minValueTrackbarCallback, (void*)&featuresDetect);
	bool result;
	while(1)
	{
		camera >> scene_image;

		result = featuresDetect.getObject(scene_image);

		cv::drawMatches(featuresDetect.m_objectImage, featuresDetect.m_objectKeypoints,
				scene_image, featuresDetect.m_sceneKeypoints,
				featuresDetect.m_goodMatches, match_image,
				cv::Scalar::all(-1), cv::Scalar::all(-1),
				std::vector<char>());

		if(result)
		{
			//object_corners[4] = featuresDetect.m_goodObjectKeypoints[featuresDetect.m_goodObjectKeypoints.size()-1].pt;
			cv::perspectiveTransform(object_corners, scene_corners, featuresDetect.m_transH);
			drawObject(match_image, scene_corners, cv::Point2f(featuresDetect.m_objectImage.cols, 0));
		}


		cv::imshow("Features Detect", match_image);
		cv::waitKey(1);
	}

	return 0;
}
