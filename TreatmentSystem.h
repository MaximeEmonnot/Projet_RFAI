#pragma once
#include <string>
#include <opencv2/core/mat.hpp>

namespace TreatmentSystem
{
    class Test {
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
        static void TestHistogramEqualization(std::string const& path);
        static void TestLeafCanny(std::string const& path);
        static void TestLeafSobel(std::string const& path);
        static void TestLeafLaplacian(std::string const& path);
        static void TestGrabCut(std::string const& path);
        static void TestNegative(std::string const& path);

        static void DisplayColorspaces(std::string const& path);
    };

    class Program
    {
    public:
        static void Run(std::string const& path);

    private:
        static void    HistogramStudy(cv::Mat const& image, std::pair<int, std::string> colorSpace);
        static cv::Mat ExtractLeaf(cv::Mat const& image);
        static cv::Mat ExtractSpots(cv::Mat const& image);
    };
}

