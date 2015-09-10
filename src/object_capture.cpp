#include <iostream>
#include "string.h"
#include "opencv2/opencv.hpp"
#include "sample_config.hpp"

int main(void)
{
	cv::VideoCapture cam(CAMERA_NUM);
	cam.set(cv::CAP_PROP_FRAME_WIDTH, CAM_FRAME_WIDTH);
	cam.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_FRAME_HEIGHT);

	cv::Mat image;
	cv::namedWindow("Image");
	while(1)
	{
		cam >> image;
		cv::imshow("Image", image);
		if (cv::waitKey(1) != -1)
			break;
	}
	std::string image_name;
	std::cout << "Image name: ";
	std::cin >> image_name;
	image_name = "image/" + image_name;
	cv::imwrite(image_name, image);
	return 0;
}
