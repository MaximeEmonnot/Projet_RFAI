#include <Color.h>

Color::Color(cv::Vec3b const& input)
: color(input)
{}

Color::Color(uint8_t red, uint8_t green, uint8_t blue)
: color(cv::Vec3b(red, green, blue))
{}

uint8_t Color::GetRed() const
{
    return color[0];
}

uint8_t Color::GetGreen() const
{
    return color[1];
}

uint8_t Color::GetBlue() const
{
    return color[2];
}

cv::Vec3b Color::GetColor() const
{
    return color;
}
