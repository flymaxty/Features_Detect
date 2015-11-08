#include "features_detect.hpp"
//#include <math.h>

FeaturesDetect::FeaturesDetect()
{
	m_nearKeypointsNumber = NEAR_KEYPOINTS_NUM;
	m_hessianThresold = HESSIAN_THRESOLD;
	m_goodMatchDistanceTimes = GOOD_MATCH_DISTANCE_TIMES;
	m_goodMatchMinValue = GOOD_MATCH_MIN_VALUE;
	m_minObjectDistance = MIN_OBJECT_DISTANCE;

	//Init Surf and SurfDescriptor
	m_surf = cv::xfeatures2d::SURF::create(HESSIAN_THRESOLD);
}

FeaturesDetect::~FeaturesDetect()
{

}

void FeaturesDetect::printInfo(std::string in_string)
{
	std::cout << in_string << std::endl;
}

void FeaturesDetect::init(std::string in_objectName, cv::Mat& in_image)
{
	m_objectName = in_objectName;
	in_image.copyTo(m_objectImage);

	//Calculate object keypoints and descriptors
	m_surf->detectAndCompute(m_objectImage, cv::noArray(), m_objectKeypoints, m_objectDescriptors);
}

void FeaturesDetect::refreshHessianThreshold(double in_value)
{
	m_surf->setHessianThreshold(in_value);
	m_surf->detectAndCompute(m_objectImage, cv::noArray(), m_objectKeypoints, m_objectDescriptors);
}

bool FeaturesDetect::getGoodMatches()
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
		printInfo("No object found!!!");
		std::cout << "minMatchDis: " << minMatchDis << std::endl;
		return false;
	}

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
		printInfo("m_goodMatches is empty");
		return false;
	}

	return true;
}

void FeaturesDetect::getEdges(cv::Mat& in_transH, std::vector<cv::Point2f>& in_edges)
{
	std::vector<cv::Point2f> ObjectEdges(5);
	ObjectEdges[0] = cv::Point2f(0,0);
	ObjectEdges[1] = cv::Point2f(m_objectImage.cols, 0);
	ObjectEdges[2] = cv::Point2f(m_objectImage.cols, m_objectImage.rows);
	ObjectEdges[3] = cv::Point2f(0, m_objectImage.rows);
	ObjectEdges[4] = cv::Point2f(m_objectImage.cols/2, m_objectImage.rows/2);

	cv::perspectiveTransform(ObjectEdges, in_edges, in_transH);
}

bool FeaturesDetect::getTransH(cv::Mat& in_transH)
{
	static std::vector<cv::Point2f> goodObjectPoints;
	static std::vector<cv::Point2f> goodScenePoints;
	static double lastDeterminant;

	goodObjectPoints.clear();
	goodScenePoints.clear();
	for(size_t i = 0; i < m_goodMatches.size(); i++)
	{
		goodObjectPoints.push_back(m_objectKeypoints[m_goodMatches[i].queryIdx].pt);
		goodScenePoints.push_back(m_sceneKeypoints[m_goodMatches[i].trainIdx].pt);
	}

	in_transH = findHomography(goodObjectPoints, goodScenePoints, cv::RANSAC);
	if(in_transH.empty())
	{
		printInfo("m_transH is empty");
		return false;
	}


	cv::Mat tMat(in_transH, cv::Rect(0, 0, 2, 2));
	double currentDeterminant = cv::determinant(tMat);
	double value = fabs(currentDeterminant - lastDeterminant);
	if(value > 0.12 && lastDeterminant != 0)
	{
		printInfo("m_transH changes too much!");
		return false;
	}
	lastDeterminant = currentDeterminant;

	return true;
}

bool FeaturesDetect::getLocation(cv::Mat& in_sceneImage, ObjectLocation& in_objectLocation, bool in_debugWindow=false)
{
	if(in_sceneImage.empty())
	{
		printInfo("in_sceneImage is empty");
		return false;
	}

	m_surf->detectAndCompute(in_sceneImage, cv::noArray(), m_sceneKeypoints, m_sceneDescriptors, false);

	if(m_sceneKeypoints.size() == 0)
	{
		printInfo("m_sceneKeypoints is empty");
		return false;
	}

	m_matcher.match(m_objectDescriptors, m_sceneDescriptors, m_matches);

	if(!getGoodMatches())
	{
		return false;
	}

	if(!getTransH(in_objectLocation.transH))
	{
		return false;
	}

	getEdges(in_objectLocation.transH, in_objectLocation.edges);

	if(in_debugWindow)
		ShowDebugWindow(in_sceneImage, in_objectLocation.edges);

	return true;
}

void FeaturesDetect::drawObject(cv::Mat& in_image, std::vector<cv::Point2f>& in_edges,
								cv::Scalar in_scalar, cv::Point2f in_offset)
{
	int thickness = 2;
	int lineType = cv::LINE_AA;
	int shift = 0;
	double t = 0;

	//Draw outlines
	cv::line(in_image, in_edges[0] + in_offset, in_edges[1] + in_offset, in_scalar, thickness, lineType, shift);
	cv::line(in_image, in_edges[1] + in_offset, in_edges[2] + in_offset, in_scalar, thickness, lineType, shift);
	cv::line(in_image, in_edges[2] + in_offset, in_edges[3] + in_offset, in_scalar, thickness, lineType, shift);
	cv::line(in_image, in_edges[3] + in_offset, in_edges[0] + in_offset, in_scalar, thickness, lineType, shift);

	//Draw center
	cv::circle(in_image, in_edges[4]+in_offset, 2, in_scalar, 2, lineType);
}

void FeaturesDetect::drawObjectName(cv::Mat& in_image, std::vector<cv::Point2f>& in_edges,
									cv::Scalar in_scalar, cv::Point2f in_offset)
{
	int baseLine = 0;
	int fontFace = cv::FONT_HERSHEY_SIMPLEX;
	double fontScale = 1.4;
	int thickness = 2;
	int lineType = cv::LINE_AA;

	cv::Size textSize;
	textSize = cv::getTextSize(m_objectName, fontFace, fontScale, thickness, &baseLine);

	double minY = 9999, maxY = 0;
	for(unsigned char i=0; i<=in_edges.size()-1; i++)
	{
		if(in_edges[i].y < minY)
			minY = in_edges[i].y;

		if(in_edges[i].y > maxY)
			maxY = in_edges[i].y;
	}

	cv::Point textCoordinate;
	textCoordinate = cv::Point(in_edges[4].x - textSize.width/2 + in_offset.x,
			minY - textSize.height + in_offset.y);
	if(textCoordinate.y - textSize.height/2 < 0)
		textCoordinate.y = maxY + textSize.height + in_offset.y + 10;

	//cv::rectangle(in_image, textCoordinate + cv::Point(0, baseLine),
	//		textCoordinate + cv::Point(textSize.width, -textSize.height),
	//         cv::Scalar(0,0,255));

	cv::putText(in_image, m_objectName, textCoordinate, fontFace, fontScale, in_scalar, thickness, lineType);
}

void FeaturesDetect::ShowDebugWindow(cv::Mat& in_seneImage, std::vector<cv::Point2f>& in_edges)
{
	cv::Mat debugImage;

	cv::drawMatches(m_objectImage, m_objectKeypoints, in_seneImage, m_sceneKeypoints,
			m_goodMatches, debugImage, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>());

	cv::Point2f offset(m_objectImage.cols, 0);
	drawObject(debugImage, in_edges, cv::Scalar(0, 255, 255), offset);
	drawObjectName(debugImage, in_edges, cv::Scalar(0, 255, 255), offset);

	cv::imshow("FeaturesDetect Debug Window", debugImage);
}
