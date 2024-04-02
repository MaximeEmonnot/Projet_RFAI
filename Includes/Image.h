#pragma once
#include <opencv2/opencv.hpp>

class Image
{
public:
    Image(std::string const& path);

    void SetRGB(int x, int y, bool color);
    bool GetRGB(int x, int y);

    void WriteToFile(std::string const& path) const;

    int GetWidth() const;
    int GetHeight() const;

    cv::Mat GetImage() const;
private:
    cv::Mat image;
};
