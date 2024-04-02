#include <Window.h>
#include <opencv2/opencv.hpp>

Window::Window(std::string const& name)
: name(name)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
}

Window::~Window()
{
    cv::waitKey(0);
}

void Window::DisplayImage(Image const& image) const
{
    cv::imshow(name, image.GetImage());
}
