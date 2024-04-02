#pragma once
#include <opencv2/opencv.hpp>

#include <Color.h>

class Image
{
public:
    Image(std::string const& path);
    ~Image() = default;
    Image(Image const& copy);
    Image& operator= (Image const& copy);
    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    void SetRGB(int x, int y, Color const& color);
    Color GetRGB(int x, int y) const;

    void WriteToFile(std::string const& path) const;

    int GetWidth() const;
    int GetHeight() const;

    cv::Mat GetImage() const;
private:
    cv::Mat image;
};
