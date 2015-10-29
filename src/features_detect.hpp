#ifndef __FEATURES_DETECT_HPP__
#define __FEATURES_DETECT_HPP__

#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

#define NEAR_KEYPOINTS_NUM		50
#define HESSIAN_THRESOLD		500
#define GOOD_MATCH_DISTANCE_TIMES 3.0
#define GOOD_MATCH_MIN_VALUE 0.02
#define MIN_OBJECT_DISTANCE 0.15

class FeaturesDetect{
public:
	struct ObjectLocation
	{
		bool isFound;
		cv::Point center;
		cv::Mat transH;
		std::vector<cv::Point2f> edges;
	};

	//Object Image location and Mat
	std::string m_objectName;
	cv::Mat m_objectImage;
	std::vector<cv::Point> m_ObjectEdges;

	//SURF
	int m_hessianThresold;
	cv::Ptr<cv::xfeatures2d::SURF> m_surf;

	//FlannBasedMatcher
	int m_goodMatchDistanceTimes;
	double m_goodMatchMinValue;
	double m_minObjectDistance;
	size_t m_nearKeypointsNumber;
	cv::FlannBasedMatcher m_matcher;
	std::vector<cv::DMatch> m_matches;
	std::vector<cv::DMatch> m_goodMatches;

	//Object value
	std::vector<cv::KeyPoint> m_objectKeypoints;
	std::vector<cv::KeyPoint> m_goodObjectKeypoints;
	cv::Mat m_objectDescriptors;

	//Scene value
	std::vector<cv::KeyPoint> m_sceneKeypoints;
	std::vector<cv::KeyPoint> m_goodSceneKeypoints;
	cv::Mat m_sceneDescriptors;
private:
	/*private value*/

public:
	FeaturesDetect();
	~FeaturesDetect();

	void init(std::string in_objectName, cv::Mat& in_image);
	void printInfo(std::string in_string);
	void refreshHessianThreshold(double in_value);
	bool getLocation(cv::Mat& in_sceneImage, ObjectLocation& in_objectLocation, bool in_debugWindow);
	void drawObject(cv::Mat& in_image, std::vector<cv::Point2f>& in_edges, cv::Scalar in_scalar, cv::Point2f in_offset=cv::Point2f(0,0));
	void drawObjectName(cv::Mat& in_image, std::vector<cv::Point2f>& in_edges, cv::Scalar in_scalar, cv::Point2f in_offset=cv::Point2f(0,0));
	void ShowDebugWindow(cv::Mat& in_sceneImage, std::vector<cv::Point2f>& in_edges);

private:
	bool getTransH(cv::Mat& in_transH);
	void getEdges(cv::Mat& in_transH, std::vector<cv::Point2f>& in_edges);
	bool getGoodMatches();
};

#endif /* __FEATURES_DETECT_HPP__ */
