#include <EngineException.h>

#include <Image.h>

int main()
{
	try
	{
		Image const image("Assets/Test.jpg");

		cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
		cv::imshow("Display Window", image.GetImage());

		cv::waitKey(0);
	}
	catch(EngineException const& e)
	{
		e.DisplayMessageBox();
	}
	return 0;
}
