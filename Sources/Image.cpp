#include <Image.h>

#include "EngineException.h"

Image::Image(std::string const& path)
: image(cv::imread(path))
{
    if(image.empty())
        throw EXCEPTION(L"OpenCV Project Exception", L"Image provided was empty !");
}

void Image::SetRGB(int x, int y, bool color)
{
}

bool Image::GetRGB(int x, int y)
{
    return false;
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
