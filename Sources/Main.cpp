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
	catch(std::exception const& e)
	{
		std::cout << e.what() << std::endl;
	}
	return 0;
}