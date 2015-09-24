#include <iostream>
#include "string.h"
#include "opencv2/opencv.hpp"
#include "sample_config.hpp"

cv::Rect selects;
bool select_flag = false;
bool selectFinish = false;
cv::Point origin;
cv::Mat image, copyImage;
void onMouse(int event, int x, int y, int,void*)
{
	if(select_flag)
	{
		selects.x = MIN(origin.x, x);
		selects.y = MIN(origin.y, y);
		selects.width = abs(x-origin.x);
		selects.height = abs(y-origin.y);
		selects &= cv::Rect(0, 0, image.cols, image.rows);
	}
	if(event == cv::EVENT_LBUTTONDOWN)
	{
		select_flag = true;
		origin = cv::Point(x,y);
		selects = cv::Rect(x,y,0,0);
	}
	else if(event == cv::EVENT_LBUTTONUP)
	{
		select_flag = false;
		selectFinish = true;
		cv::Rect crop(selects.x, selects.y, selects.width, selects.height);
		std::string image_name;
		std::cout << "Image name: ";
		std::cin >> image_name;
		image_name = "image/" + image_name;
		cv::imwrite(image_name, copyImage(crop));
		exit(0);
	}
}

int main(void)
{
	cv::VideoCapture cam(CAMERA_NUM);
	cam.set(cv::CAP_PROP_FRAME_WIDTH, CAM_FRAME_WIDTH);
	cam.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_FRAME_HEIGHT);

	//cv::Mat image;
	cv::namedWindow("Image");

	cv::setMouseCallback("Image", onMouse, NULL);

	bool refresh = true;
	while(1)
	{
		if(refresh)
			cam >> image;
		else
			copyImage.copyTo(image);

		cv::rectangle(image,selects,cv::Scalar(255,0,0),3,8,0);

		cv::imshow("Image", image);

		if (cv::waitKey(1) != -1)
		{
			image.copyTo(copyImage);
			refresh = false;
		}
	}
	return 0;
}
