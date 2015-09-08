#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "object_detect_config.hpp"

using namespace std;
using namespace cv;

void draw_object(InputOutputArray img, vector<Point2f> &pt,Point2f offset_point)
{
	Scalar color(0, 0, 255);
	int thickness = 1;
	int lineType = cv::LINE_8;
	int shift = 0;

	//边框
	line(img, pt[0] + offset_point, pt[1] + offset_point, color, thickness, lineType, shift);
	line(img, pt[1] + offset_point, pt[2] + offset_point, color, thickness, lineType, shift);
	line(img, pt[2] + offset_point, pt[3] + offset_point, color, thickness, lineType, shift);
	line(img, pt[3] + offset_point, pt[0] + offset_point, color, thickness, lineType, shift);

	//单点
	cv::circle(img, pt[4]+offset_point, 12, Scalar(0, 0, 255), cv::FILLED);
}

int main()
{
	//图像读取
	namedWindow("Match");
	Mat image_object = imread(OBJECT_IMG);
	Mat image_sense = imread(SCENE_IMG);

	//SURF初始化
	Mat descriptorsA, descriptorsB;
	std::vector<KeyPoint> keypointsA, keypointsB;
	Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(HESSIAN_THRESOLD);
	Ptr<xfeatures2d::SURF> surf_ext = xfeatures2d::SurfDescriptorExtractor::create();
	//特征点匹配
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;

	//特征点检测
	surf->detect(image_object, keypointsA);
	surf->detect(image_sense, keypointsB);
	cout << keypointsA.size() << endl;
	//计算SURF特征描数子
	surf_ext->compute(image_object, keypointsA, descriptorsA);
	surf_ext->compute(image_sense, keypointsB, descriptorsB);
	//特征点匹配
	matcher.match(descriptorsA, descriptorsB, matches);

	Mat pre_descriptorsB;
	std::vector<KeyPoint> pre_keypointsB;
	std::vector<cv::DMatch> pre_matches;
	size_t min;
	double a1=0, a2=9999;
	double match_minDis = 9999, dis;
	int match_min;
	double total_x=0, total_y=0;
	unsigned char i_size;
	if(matches.size() > PRE_FILTER_NUM)
	{
		for ( size_t i=0; i<matches.size(); i++ )
		{
			if ( matches[i].distance < match_minDis )
			{
				match_minDis = matches[i].distance;
				match_min = i;
			}
		}

		for (unsigned char j=0; j<PRE_FILTER_NUM; j++ )
		{
			for(size_t i=0; i<keypointsB.size(); i++)
			{
				dis = cv::norm(keypointsB[i].pt - keypointsB[matches[match_min].trainIdx].pt);
				if ( dis > a1 && dis < a2)
				{
					a2 = dis;
					min = i;
				}
			}

			cout << a2 << endl;
			pre_keypointsB.push_back(keypointsB[min]);
			a1 = a2;
			a2 = 9999;
		}

		cout << pre_keypointsB.size();
		surf_ext->compute(image_sense, pre_keypointsB, pre_descriptorsB);
		matcher.match(descriptorsA, pre_descriptorsB, pre_matches);
	}

	//Ransac检测
	int ptCount = (int)pre_matches.size();
	cout << "pre" << ptCount << endl;
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	Point2f pt;
	for (int i=0; i<ptCount; i++)
	{
	     pt = keypointsA[pre_matches[i].queryIdx].pt;
	     p1.at<float>(i, 0) = pt.x;
	     p1.at<float>(i, 1) = pt.y;

	     pt = pre_keypointsB[pre_matches[i].trainIdx].pt;
	     p2.at<float>(i, 0) = pt.x;
	     p2.at<float>(i, 1) = pt.y;
	}

	//Ransac筛点
	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;
	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

	int OutlinerCount = 0;
	for (int i=0; i<ptCount; i++)
	{
	     if (m_RANSACStatus[i] == 0) // 状态为0表示野点
	     {
	          OutlinerCount++;
	     }
	}

	// 计算内点
	vector<Point2f> m_LeftInlier;
	vector<Point2f> m_RightInlier;
	vector<DMatch> goodMatches;
	// 上面三个变量用于保存内点和匹配关系
	int InlinerCount = ptCount - OutlinerCount;
	goodMatches.resize(InlinerCount);
	m_LeftInlier.resize(InlinerCount);
	m_RightInlier.resize(InlinerCount);
	InlinerCount = 0;
	for (int i=0; i<ptCount; i++)
	{
	     if (m_RANSACStatus[i] != 0)
	     {
	          m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
	          m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
	          m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
	          m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
	          goodMatches[InlinerCount].queryIdx = InlinerCount;
	          goodMatches[InlinerCount].trainIdx = InlinerCount;
	          InlinerCount++;
	     }
	}

	// 把内点转换为drawMatches可以使用的格式
	vector<KeyPoint> goodkeypointsA(InlinerCount);
	vector<KeyPoint> goodkeypointsB(InlinerCount);
	KeyPoint::convert(m_LeftInlier, goodkeypointsA);
	KeyPoint::convert(m_RightInlier, goodkeypointsB);

	Mat H;
	H = findHomography(m_LeftInlier, m_RightInlier, cv::RANSAC);

	vector<Point2f> object_corners(4), scense_corners(4);
	object_corners[0] = Point(0,0);
	object_corners[1] = Point(image_object.cols, 0);
	object_corners[2] = Point(image_object.cols, image_object.rows);
	object_corners[3] = Point(0, image_object.rows);
	cv::perspectiveTransform(object_corners, scense_corners, H);

	cv:Mat image_match;
	cv::drawMatches(image_object, goodkeypointsA,
			image_sense, goodkeypointsB,
			goodMatches, image_match,
			Scalar::all(-1), Scalar::all(-1),
			vector<char>(),
			cv::DrawMatchesFlags::DEFAULT);

	imshow("Match", image_match);
	waitKey(0);
	return 0;
}
