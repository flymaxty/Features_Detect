
#define __FEATURES_DETECT_HPP__

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

#define NEAR_KEYPOINTS_NUM		30
#define HESSIAN_THRESOLD		3000

class CfeaturesDetect{
public:
	//Object image Mat
	cv::Mat m_objectImage;

	//Object transformation H
	cv::Mat m_transH;

	//SURF
	cv::Ptr<cv::xfeatures2d::SURF> m_surf;
	cv::Ptr<cv::xfeatures2d::SURF> m_surfExtractor;

	//FlannBasedMatcher
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
	void getObject(cv::Mat in_sceneImage);
	void getGoodMatches();
};

#endif /* __FEATURES_DETECT_HPP__ */