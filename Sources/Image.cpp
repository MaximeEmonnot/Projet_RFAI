#include <Image.h>

#include "EngineException.h"

Image::Image(std::string const& path)
: image(cv::imread(path))
{
    if(image.empty())
        throw EXCEPTION(L"OpenCV Project Exception", L"Image provided was empty !");
}

void Image::SetRGB(int x, int y, Color const& color)
{
    image.at<cv::Vec3b>(x, y) = color.GetColor();
}

Color Image::GetRGB(int x, int y) const
{
    return Color(image.at<cv::Vec3b>(x, y));
}

void Image::WriteToFile(std::string const& path) const
{
    cv::imwrite(path, image);
}

int Image::GetWidth() const
{
    return image.cols;
}

int Image::GetHeight() const
{
    return image.rows;
}

cv::Mat Image::GetImage() const
{
    return image;
}
