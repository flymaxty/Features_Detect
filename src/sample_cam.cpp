#include <iostream>
#include <string>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "sample_config.hpp"
#include "features_detect.hpp"

void hessianTrackbarCallback(int thres, void* in_featuresDetect)
{
	FeaturesDetect* featuresDetect =  (FeaturesDetect*) in_featuresDetect;
	featuresDetect->refreshHessianThreshold((double)thres);
}

void goodMatchMinValueTrackbarCallback(int thres, void* in_featuresDetect)
{
	FeaturesDetect* featuresDetect =  (FeaturesDetect*) in_featuresDetect;
	featuresDetect->m_goodMatchMinValue = thres / 100.0;
}

void minObjectDistanceTrackbarCallback(int thres, void* in_featuresDetect)
{
	FeaturesDetect* featuresDetect =  (FeaturesDetect*) in_featuresDetect;
	featuresDetect->m_minObjectDistance = thres / 100.0;
}

std::string getFileName(const std::string& in_string)
{
#ifdef _MSC_VER
	unsigned short begin = in_string.find_last_of('\\');
#else
	unsigned short begin = in_string.find_last_of('/');
#endif

	unsigned short end = in_string.find_last_of('.');
	return in_string.substr(begin+1, end-begin-1);
}

int main(void)
{
	cv::Mat objectImage = cv::imread(OBJECT_IMG);
	cv::Mat sceneImage, outputImage;
	FeaturesDetect::ObjectLocation objectLocation;

	cv::VideoCapture camera(CAMERA_NUM);
	camera.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_FRAME_HEIGHT);
	camera.set(cv::CAP_PROP_FRAME_WIDTH, CAM_FRAME_WIDTH);
	cv::namedWindow("Features Detect");

	FeaturesDetect featuresDetect;
	featuresDetect.init(getFileName(OBJECT_IMG), objectImage);

	cv::namedWindow("Config");
	int initGoodMatchMinValue = featuresDetect.m_goodMatchMinValue * 100.0;
	int initMinObjectDistance = featuresDetect.m_minObjectDistance * 100.0;
	cv::createTrackbar("Hessian Thresold", "Config", &featuresDetect.m_hessianThresold,
						10000, hessianTrackbarCallback, (void*)&featuresDetect);
	cv::createTrackbar("Min Object Distance", "Config", &initMinObjectDistance,
						100, minObjectDistanceTrackbarCallback, (void*)&featuresDetect);
	cv::createTrackbar("Good Match Distance Times", "Config", &featuresDetect.m_goodMatchDistanceTimes,
						10, NULL);
	cv::createTrackbar("Good Match Min Value(100 times)", "Config", &initGoodMatchMinValue,
						50, goodMatchMinValueTrackbarCallback, (void*)&featuresDetect);

	bool result;
	while(1)
	{
		camera >> sceneImage;
		if (sceneImage.empty())
		{
			std::cout << "End!" << std::endl;
			return 0;
		}
		sceneImage.copyTo(outputImage);
		//cv::cvtColor(scene_image, scene_image, cv::COLOR_RGB2GRAY);
		//cv::equalizeHist(scene_Image, scene_Image);

		result = featuresDetect.getLocation(sceneImage, objectLocation, false);

		if(result)
		{
			featuresDetect.drawObject(outputImage, objectLocation.edges, cv::Scalar(0, 0, 255));
			featuresDetect.drawObjectName(outputImage, objectLocation.edges, cv::Scalar(0, 0, 255));
		}
		else
		{
			cv::putText(outputImage, "No Object Found!!", cv::Point2f(0, 30),
			cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
		}

		cv::imshow("Features Detect", outputImage);

		cv::waitKey(1);
	}
	return 0;
}
