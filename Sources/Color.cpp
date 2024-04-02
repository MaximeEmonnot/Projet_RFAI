#include <Color.h>

Color::Color(cv::Vec3b const& input)
: color(input)
{}

Color::Color(uint8_t red, uint8_t green, uint8_t blue)
: color(cv::Vec3b(red, green, blue))
{}

cv::Vec3b Color::GetColor() const
{
    return color;
}
