#include "features_detect.hpp"

CfeaturesDetect::CfeaturesDetect()
{
	//Init Surf and SurfDescriptor
	m_surf = cv::xfeatures2d::SURF::create(HESSIAN_THRESOLD);
	m_surfExtractor = cv::xfeatures2d::SurfDescriptorExtractor::create();
}

CfeaturesDetect::~CfeaturesDetect()
{

}

void CfeaturesDetect::init()
{
	//Calculate object keypoints and descriptors
	m_surf->detect(m_objectImage, m_objectKeypoints);
	m_surfExtractor->compute(m_objectImage, m_objectKeypoints, m_objectDescriptors);
}

void CfeaturesDetect::getGoodMatches()
{
	m_goodMatches.clear();
	m_goodObjectKeypoints.clear();
	m_goodSceneKeypoints.clear();

	//Calculate closest match
	double minMatchDis = 9999;
	size_t minMatchIndex = 0;
	for ( size_t i=0; i<m_matches.size(); i++ )
	{
		if ( m_matches[i].distance < minMatchDis )
		{
			minMatchDis = m_matches[i].distance;
			minMatchIndex = i;
		}
	}

	double a1=0, a2=9999, dis;
	size_t min;
	for (size_t j=0; j<NEAR_KEYPOINTS_NUM; j++ )
	{
		for(size_t i=0; i<m_matches.size(); i++)
		{
			dis = cv::norm(m_sceneKeypoints[m_matches[i].trainIdx].pt - m_sceneKeypoints[m_matches[minMatchIndex].trainIdx].pt);
			if ( dis > a1 && dis < a2)
			{
				a2 = dis;
				min = i;
			}
		}
		m_goodMatches.push_back(m_matches[min]);
		m_goodObjectKeypoints.push_back(m_objectKeypoints[m_matches[min].queryIdx]);
		m_goodSceneKeypoints.push_back(m_sceneKeypoints[m_matches[min].trainIdx]);

		a1 = a2;
		a2 = 9999;
	}
}

void CfeaturesDetect::getObject(cv::Mat in_sceneImage)
{
	m_surf->detect(in_sceneImage, m_sceneKeypoints);
	std::cout << "m_sceneKeypoints: " << m_sceneKeypoints.size() << std::endl;
	m_surfExtractor->compute(in_sceneImage, m_sceneKeypoints, m_sceneDescriptors);

	m_matcher.match(m_objectDescriptors, m_sceneDescriptors, m_matches);

	getGoodMatches();


	std::vector<cv::Point2f> goodObjectPoints;
	std::vector<cv::Point2f> goodScenePoints;
	for(size_t i = 0; i < m_goodMatches.size(); i++)
	{
		goodObjectPoints.push_back(m_objectKeypoints[m_goodMatches[i].queryIdx].pt);
		goodScenePoints.push_back(m_sceneKeypoints[m_goodMatches[i].trainIdx].pt);
	}
	m_transH = findHomography(goodObjectPoints, goodScenePoints, cv::RANSAC);
}