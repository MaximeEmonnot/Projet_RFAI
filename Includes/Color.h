#pragma once
#include <opencv2/opencv.hpp>

class Color
{
public:
    Color(cv::Vec3b const& input);
    Color(uint8_t red, uint8_t green, uint8_t blue);

    cv::Vec3b GetColor() const;

private:
    cv::Vec3b color;
};