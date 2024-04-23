#pragma once
#include <memory>
#include <string>

class TreatmentSystem
{
public:
    TreatmentSystem()                                   = default;
    TreatmentSystem(TreatmentSystem const&)             = delete;
    TreatmentSystem& operator= (TreatmentSystem const&) = delete;
    TreatmentSystem(TreatmentSystem&&)                  = delete;
    TreatmentSystem& operator= (TreatmentSystem&&)      = delete;

    static TreatmentSystem& GetInstance();

    void RunTests(std::string const& path);
protected:
    void TestHistogram(std::string const& path);
    void TestHistogramGrayscale(std::string const& path);
    void TestContour(std::string const& path);
    void TestBlobDetection(std::string const& path);
    void TestEdgeDetection(std::string const& path);
    void TestThresholding(std::string const& path);
    void TestColorspaces(std::string const& path);

private:
    static std::unique_ptr<TreatmentSystem> pInstance;
};

