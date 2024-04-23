#pragma once
#include <string>

class TreatmentSystem
{
public:
    static void RunTests(std::string const& path);

private:
    static void TestHistogram(std::string const& path);
    static void TestHistogramGrayscale(std::string const& path);
    static void TestContour(std::string const& path);
    static void TestBlobDetection(std::string const& path);
    static void TestEdgeDetection(std::string const& path);
    static void TestThresholding(std::string const& path);
    static void TestColorspaces(std::string const& path);
};

