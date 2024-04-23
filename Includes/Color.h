#pragma once
#include <opencv2/opencv.hpp>

enum ColorChannel : uint8_t
{
	NONE  = 0,
    RED   = 1,
    GREEN = 2,
    BLUE  = 4
};

class Color
{
public:
    Color(cv::Vec3b const& input);
    Color(uint8_t red, uint8_t green, uint8_t blue);

    uint8_t GetRed() const;
    uint8_t GetGreen() const;
    uint8_t GetBlue() const;

    cv::Vec3b GetColor() const;

private:
    cv::Vec3b color;
};