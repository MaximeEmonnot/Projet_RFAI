#pragma once
#include "Image.h"

class Transformations
{
public:
    /*
     * Source : https://en.wikipedia.org/wiki/Grayscale
     * At "Luma coding in video systems"
     */
    static Image ToGrayScale(Image const& image)
    {
        Image output = image;

        for(int x = 0; x < image.GetWidth(); x++)
        {
	        for(int y = 0; y < image.GetHeight(); y++)
	        {
                Color   const color     = image.GetRGB(x, y);
                uint8_t const grayValue = static_cast<uint8_t>(0.299 * color.GetRed()) + static_cast<uint8_t>(0.587 * color.GetGreen()) + static_cast<uint8_t>(0.114 * color.GetBlue());
                output.SetRGB(x, y, { grayValue, grayValue, grayValue});
	        }
        }

        return output;
    }

    static Image MaskChannel(Image const& image, int channel)
    {
        Image output = image;

        for(int x = 0; x < image.GetWidth(); x++)
        {
	        for(int y = 0; y < image.GetHeight(); y++)
	        {
                Color const color = image.GetRGB(x, y);

                output.SetRGB(x, y, { (channel & RED)    == RED   ? color.GetRed()   : static_cast<uint8_t>(0)
                                            , (channel & GREEN) == GREEN ? color.GetGreen() : static_cast<uint8_t>(0)
                                            , (channel & BLUE)   == BLUE  ? color.GetBlue()  : static_cast<uint8_t>(0)});
	        }
        }

        return output;
    }
};
