#ifndef __FEATURES_DETECT_HPP__
#define __FEATURES_DETECT_HPP__

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

#define NEAR_KEYPOINTS_NUM		50
#define HESSIAN_THRESOLD		300
#define GOOD_MATCH_DISTANCE_TIMES 3.0
#define GOOD_MATCH_MIN_VALUE 0.02
#define MIN_OBJECT_DISTANCE 0.2

class CfeaturesDetect{
public:
	//Object image Mat
	cv::Mat m_objectImage;

	//Object transformation H
	cv::Mat m_transH;

	//SURF
	int m_hessianThresold;
	cv::Ptr<cv::xfeatures2d::SURF> m_surf;
	cv::Ptr<cv::xfeatures2d::SURF> m_surfExtractor;

	//FlannBasedMatcher
	int m_goodMatchDistanceTimes;
	double m_goodMatchMinValue;
	double m_minObjectDistance;
	size_t m_nearKeypointsNumber;
	cv::FlannBasedMatcher m_matcher;
	std::vector<cv::DMatch> m_matches;
	std::vector<cv::DMatch> m_goodMatches;
	//size_t bestMatchIndex;

	//Object value
	std::vector<cv::KeyPoint> m_objectKeypoints;
	std::vector<cv::KeyPoint> m_goodObjectKeypoints;
	cv::Mat m_objectDescriptors;

	//Scene value
	std::vector<cv::KeyPoint> m_sceneKeypoints;
	std::vector<cv::KeyPoint> m_goodSceneKeypoints;
	cv::Mat m_sceneDescriptors;
public:
	CfeaturesDetect();
	~CfeaturesDetect();

	void init();
	void clearMatch();
	bool getObject(cv::Mat in_sceneImage);
	void getGoodMatches();
	bool getGoodMatchesA();
};

#endif /* __FEATURES_DETECT_HPP__ */
