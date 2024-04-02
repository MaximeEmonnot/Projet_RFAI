#include <Image.h>

#include "EngineException.h"

Image::Image(std::string const& path)
: image(cv::imread(path))
{
    if(image.empty())
        throw EXCEPTION(L"OpenCV Project Exception", L"Image provided was empty !");
}

Image::Image(Image const& copy)
{
    *this = copy;
}

Image& Image::operator=(Image const& copy)
{
    image = copy.image.clone();

    return *this;
}

void Image::SetRGB(int x, int y, Color const& color)
{
    CheckCoordinates(x, y);
    image.at<cv::Vec3b>(x, y) = color.GetColor();
}

Color Image::GetRGB(int x, int y) const
{
    CheckCoordinates(x, y);
    return { image.at<cv::Vec3b>(x, y) };
}

void Image::WriteToFile(std::string const& path) const
{
    cv::imwrite(path, image);
}

int Image::GetWidth() const
{
    return image.rows;
}

int Image::GetHeight() const
{
    return image.cols;
}

cv::Mat Image::GetImage() const
{
    return image;
}

void Image::CheckCoordinates(int x, int y) const
{
    if(x >= GetWidth())
        throw EXCEPTION(L"OpenCV Project Exception", L"X Coordinate is out of bounds!");
    if(y >= GetHeight())
        throw EXCEPTION(L"OpenCV Project Exception", L"Y Coordinate is out of bounds!");
}
