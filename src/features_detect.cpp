#include "features_detect.hpp"

CfeaturesDetect::CfeaturesDetect()
{
	m_nearKeypointsNumber = NEAR_KEYPOINTS_NUM;
	m_hessianThresold = HESSIAN_THRESOLD;
	m_goodMatchDistanceTimes = GOOD_MATCH_DISTANCE_TIMES;
	m_goodMatchMinValue = GOOD_MATCH_MIN_VALUE;
	m_minObjectDistance = MIN_OBJECT_DISTANCE;

	//Init Surf and SurfDescriptor
	m_surf = cv::xfeatures2d::SURF::create(HESSIAN_THRESOLD);
}

CfeaturesDetect::~CfeaturesDetect()
{

}

void CfeaturesDetect::init()
{
	//Calculate object keypoints and descriptors
	m_surf->detectAndCompute(m_objectImage, cv::noArray(), m_objectKeypoints, m_objectDescriptors);
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

	m_goodMatches.push_back(m_matches[minMatchIndex]);
	m_goodObjectKeypoints.push_back(m_objectKeypoints[m_matches[minMatchIndex].queryIdx]);
	m_goodSceneKeypoints.push_back(m_sceneKeypoints[m_matches[minMatchIndex].trainIdx]);
}

bool CfeaturesDetect::getGoodMatchesA()
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
	if(minMatchDis > m_minObjectDistance)
	{
		std::cout << "No object found!!!" << std::endl;
		std::cout << "minMatchDis: " << minMatchDis << std::endl;
		return false;
	}
/*
	if(m_goodMatchDistanceTimes * minMatchDis > m_goodMatchMinValue)
	{
		std::cout << "use m_goodMatchDistanceTimes" << std::endl;
	}
	else
	{
		std::cout << "use m_goodMatchMinValue" << std::endl;
	}
*/
	double maxDistance = std::max(m_goodMatchDistanceTimes * minMatchDis, m_goodMatchMinValue);
	for (size_t i=0; i<m_matches.size(); i++ )
	{
		if(m_matches[i].distance <= maxDistance)
		{
			m_goodMatches.push_back(m_matches[i]);
			m_goodObjectKeypoints.push_back(m_objectKeypoints[m_matches[i].queryIdx]);
			m_goodSceneKeypoints.push_back(m_sceneKeypoints[m_matches[i].trainIdx]);
		}
	}

	if(m_goodMatches.size() == 0)
	{
		std::cout << "m_goodMatches is empty" << std::endl;
		return false;
	}
	else
		std::cout << "m_goodMatches: " << m_goodMatches.size() << std::endl;

	return true;
}

bool CfeaturesDetect::getObject(cv::Mat in_sceneImage)
{
	if(in_sceneImage.empty())
	{
		std::cout << "in_sceneImage is empty" << std::endl;
		return false;
	}

	m_surf->detectAndCompute(in_sceneImage, cv::noArray(), m_sceneKeypoints, m_sceneDescriptors, false);

	if(m_sceneKeypoints.size() == 0)
	{
		std::cout << "m_sceneKeypoints is empty" << std::endl;
		return false;
	}

	m_matcher.match(m_objectDescriptors, m_sceneDescriptors, m_matches);

	if(!getGoodMatchesA())
	{
		return false;
	}

	static std::vector<cv::Point2f> goodObjectPoints;
	static std::vector<cv::Point2f> goodScenePoints;
	goodObjectPoints.clear();
	goodScenePoints.clear();
	for(size_t i = 0; i < m_goodMatches.size(); i++)
	{
		goodObjectPoints.push_back(m_objectKeypoints[m_goodMatches[i].queryIdx].pt);
		goodScenePoints.push_back(m_sceneKeypoints[m_goodMatches[i].trainIdx].pt);
	}

	m_transH = findHomography(goodObjectPoints, goodScenePoints, cv::RANSAC);
	if(m_transH.empty())
	{
		std::cout << "m_transH is empty" << std::endl;
		return false;
	}

	return true;
}
