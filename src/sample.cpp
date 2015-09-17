#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "sample_config.hpp"
#include "features_detect.hpp"

void drawObject(cv::Mat img, std::vector<cv::Point2f> &pt, cv::Point2f offset_point)
{
	cv::Scalar color(0, 0, 255);
	int thickness = 3;
	int lineType = cv::LINE_8;
	int shift = 0;

	//边框
	cv::line(img, pt[0] + offset_point, pt[1] + offset_point, color, thickness, lineType, shift);
	cv::line(img, pt[1] + offset_point, pt[2] + offset_point, color, thickness, lineType, shift);
	cv::line(img, pt[2] + offset_point, pt[3] + offset_point, color, thickness, lineType, shift);
	cv::line(img, pt[3] + offset_point, pt[0] + offset_point, color, thickness, lineType, shift);

	//单点
	cv::circle(img, pt[4]+offset_point, 3, color, cv::FILLED);
}

void hessianTrackbarCallback(int thres, void* in_featuresDetect)
{
	CfeaturesDetect* featuresDetect =  (CfeaturesDetect*) in_featuresDetect;
	featuresDetect->m_surf->setHessianThreshold(thres);
	featuresDetect->m_goodMatchMinValue = thres;
}

void minValueTrackbarCallback(int thres, void* in_featuresDetect)
{
	CfeaturesDetect* featuresDetect =  (CfeaturesDetect*) in_featuresDetect;
	featuresDetect->m_goodMatchMinValue = thres / 100.0;
}

int main(void)
{
	cv::Mat object_image = cv::imread(OBJECT_IMG);
	cv::Mat scene_image = cv::imread(SCENE_IMG);
	cv::Mat match_image;

	cv::namedWindow("Features Detect");
	cv::namedWindow("Config");

	CfeaturesDetect featuresDetect;
	featuresDetect.m_objectImage = object_image;
	featuresDetect.init();

	std::vector<cv::Point2f> object_corners(5), scene_corners(5);
	object_corners[0] = cv::Point2f(0,0);
	object_corners[1] = cv::Point2f(object_image.cols, 0);
	object_corners[2] = cv::Point2f(object_image.cols, object_image.rows);
	object_corners[3] = cv::Point2f(0, object_image.rows);

	int initMinValue = featuresDetect.m_goodMatchMinValue * 100.0;
	cv::createTrackbar("Hessian Thresold", "Config", &featuresDetect.m_hessianThresold, 10000, hessianTrackbarCallback, (void*)&featuresDetect);
	cv::createTrackbar("Good Match Distance Times", "Config", &featuresDetect.m_goodMatchDistanceTimes, 10, NULL);
	cv::createTrackbar("Good Match Min Value(100 times)", "Config", &initMinValue, 50, minValueTrackbarCallback, (void*)&featuresDetect);

	while(1)
	{
		featuresDetect.getObject(scene_image);
		object_corners[4] = featuresDetect.m_goodObjectKeypoints[featuresDetect.m_goodObjectKeypoints.size()-1].pt;
		cv::perspectiveTransform(object_corners, scene_corners, featuresDetect.m_transH);

		cv::drawMatches(featuresDetect.m_objectImage, featuresDetect.m_objectKeypoints,
				scene_image, featuresDetect.m_sceneKeypoints,
				featuresDetect.m_goodMatches, match_image,
				cv::Scalar::all(-1), cv::Scalar::all(-1),
				std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
		drawObject(match_image, scene_corners, cv::Point2f(featuresDetect.m_objectImage.cols, 0));

		cv::imshow("Features Detect", match_image);
		cv::waitKey(1);
	}

	return 0;
}
