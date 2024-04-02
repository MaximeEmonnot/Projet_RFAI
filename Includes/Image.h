#pragma once
#include <opencv2/opencv.hpp>

#include <Color.h>

class Image
{
public:
    Image(std::string const& path);

    void SetRGB(int x, int y, Color const& color);
    Color GetRGB(int x, int y) const;

    void WriteToFile(std::string const& path) const;

    int GetWidth() const;
    int GetHeight() const;

    cv::Mat GetImage() const;
private:
    cv::Mat image;
};
