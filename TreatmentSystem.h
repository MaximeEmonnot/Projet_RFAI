#pragma once
#include <string>

class TreatmentSystem
{
public:
    static void RunTests(std::string const& path);

private:
    static void TestHistogram(std::string const& path);
    static void TestHistogramGrayscale(std::string const& path);
    static void TestHistogramHSV(std::string const& path);
    static void TestHistogramLab(std::string const& path);
    static void TestContour(std::string const& path);
    static void TestBlobDetection(std::string const& path);
    static void TestEdgeDetection(std::string const& path);
    static void TestThresholding(std::string const& path);
    static void TestColorspaces(std::string const& path);
    static void TestMedianBlur(std::string const& path);
    static void TestORB(std::string const& path);
    static void TestSegmentation(std::string const& path);
    static void TestHSVSaturationIdea(std::string const& path);
    static void TestLabDarkenIdea(std::string const& path);
    static void TestContourDetection(std::string const& path);
    static void TestBackgroundSubtractor(std::string const& path);
};

