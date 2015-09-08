#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "object_detect_config.hpp"

using namespace std;
using namespace cv;

int main()
{
	namedWindow("Match");
	Mat image_scene, image_match;
	Mat image_object = imread(OBJECT_IMG);

	cv::VideoCapture cam;
	cam.open(0);
	cam.set(CAP_PROP_FRAME_WIDTH, 1024);
	cam.set(CAP_PROP_FRAME_HEIGHT, 768);

	Mat descriptorsA, descriptorsB;
	std::vector<KeyPoint> keypointsA, keypointsB;
	Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(HESSIAN_THRESOLD);
	Ptr<xfeatures2d::SURF> surf_ext = xfeatures2d::SurfDescriptorExtractor::create();

	double match_minDis = 9999;
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;

	Mat Trans_H;
	vector<Point2f> object_corners(5), scense_corners(5);
	object_corners[0] = Point(0,0);
	object_corners[1] = Point(image_object.cols, 0);
	object_corners[2] = Point(image_object.cols, image_object.rows);
	object_corners[3] = Point(0, image_object.rows);
	object_corners[4] = Point(image_object.cols/2, image_object.rows/2);

	surf->detect(image_object, keypointsA);
	surf_ext->compute(image_object, keypointsA, descriptorsA);

	Point2f pt;
	Mat m_Fundamental;
	int InlinerCount = 0;
	int OutlinerCount = 0;
	vector<uchar> m_RANSACStatus;
	vector<Point2f> m_LeftInlier;
	vector<Point2f> m_RightInlier;
	vector<DMatch> goodMatches;
	vector<KeyPoint> goodkeypointsA;
	vector<KeyPoint> goodkeypointsB;

	while(1)
	{
		cam >> image_scene;
		keypointsB.clear();
		surf->detect(image_scene, keypointsB);
		if(keypointsB.size() == 0)
		{
			cout << "No keypoints!" << endl;
			continue;
		}

		surf_ext->compute(image_scene, keypointsB, descriptorsB);

		matches.clear();
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

				//cout << a2 << endl;
				pre_keypointsB.push_back(keypointsB[min]);
				a1 = a2;
				a2 = 9999;
			}

			surf_ext->compute(image_scene, pre_keypointsB, pre_descriptorsB);
			matcher.match(descriptorsA, pre_descriptorsB, pre_matches);
		}
		else
		{
			cout << "lalal" << endl;
		}

		//Ransac检测
		int ptCount = (int)pre_matches.size();
		Mat p1(ptCount, 2, CV_32F);
		Mat p2(ptCount, 2, CV_32F);

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
		m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

		OutlinerCount = 0;
		for (int i=0; i<ptCount; i++)
		{
		     if (m_RANSACStatus[i] == 0) // 状态为0表示野点
		     {
		          OutlinerCount++;
		     }
		}

		// 计算内点
		goodMatches.clear();
		m_LeftInlier.clear();
		m_RightInlier.clear();
		InlinerCount = ptCount - OutlinerCount;
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
		goodkeypointsA.resize(InlinerCount);
		goodkeypointsB.resize(InlinerCount);
		KeyPoint::convert(m_LeftInlier, goodkeypointsA);
		KeyPoint::convert(m_RightInlier, goodkeypointsB);

		/*
		match_minDis = 9999;
		goodMatches.clear();
		goodkeypointsA.clear();
		goodkeypointsB.clear();
		for ( size_t i=0; i<matches.size(); i++ )
		{
			if ( matches[i].distance < match_minDis )
				match_minDis = matches[i].distance;
		}

		for ( size_t i=0; i<matches.size(); i++ )
		{
			if (matches[i].distance < 3*match_minDis)
			{
				goodMatches.push_back(matches[i]);
				goodkeypointsA.push_back(keypointsA[matches[i].queryIdx].pt);
				goodkeypointsB.push_back(keypointsB[matches[i].trainIdx].pt);
			}
		}
		*/

		cv::drawMatches(image_object, goodkeypointsA,
				image_scene, goodkeypointsB,
				goodMatches, image_match,
				Scalar::all(-1), Scalar::all(-1),
				vector<char>(), cv::DrawMatchesFlags::DEFAULT);
		Trans_H = findHomography(m_LeftInlier, m_RightInlier, cv::RANSAC);
		if(!Trans_H.empty())
		{
			cv::perspectiveTransform(object_corners, scense_corners, Trans_H);

			line(image_match, scense_corners[0] + Point2f(image_object.cols, 0),
					scense_corners[1] + Point2f(image_object.cols, 0), Scalar(0, 255, 0), 3 );
			line(image_match, scense_corners[1] + Point2f(image_object.cols, 0),
					scense_corners[2] + Point2f(image_object.cols, 0), Scalar(0, 255, 0), 3 );
			line(image_match, scense_corners[2] + Point2f(image_object.cols, 0),
					scense_corners[3] + Point2f(image_object.cols, 0), Scalar(0, 255, 0), 3 );
			line(image_match, scense_corners[3] + Point2f(image_object.cols, 0),
					scense_corners[0] + Point2f(image_object.cols, 0), Scalar(0, 255, 0), 3 );
			cv::circle(image_match, Point2f(image_object.cols, 0) + keypointsB[matches[match_min].trainIdx].pt, 12, Scalar(0, 0, 255), cv::FILLED);
		}

		imshow("Match", image_match);
		waitKey(1);
	}

	return 0;
}
