#include <Windows.h>

#include "../Includes/EngineException.h"
#include "opencv2/opencv.hpp"

int main()
{
	try
	{
		cv::Mat image = cv::imread("Assets/Test.jpg", cv::IMREAD_COLOR);

		cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
		cv::imshow("Display Window", image);

		cv::waitKey(0);
	}
	catch(EngineException const& e)
	{
		e.DisplayMessageBox();
	}
	return 0;
}