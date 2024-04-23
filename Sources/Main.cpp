#include <EngineException.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

// FONCTIONS TESTS DE TRAITEMENT

void TestHistogram(std::string const& path)
{
    cv::Mat src = cv::imread(path);

	std::vector<cv::Mat> bgr_planes;
    split(src, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for(int i = 1; i < histSize; i++) {
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
    }

	imshow("Source image", src);
    imshow("calcHist Demo", histImage);

	cv::imwrite("Outputs/Histogram.png", histImage);
	cv::waitKey();
}

void TestHistogramGrayscale(std::string const& path)
{
	cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };

	cv::Mat hist;
	cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);

	int hist_w = 512, hist_h = 400;
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

	for(int i = 0; i < histSize; i++) 
	{
		float binVal = hist.at<float>(i);
		int intensity = static_cast<int>(binVal * hist_h / 255);
		cv::line(histImage, cv::Point(i, hist_h), cv::Point(i, hist_h - intensity), cv::Scalar(255, 255, 255));
	}

	imshow("Source image", src);
	imshow("calcHist Demo", histImage);

	cv::waitKey();
}

void TestContour(std::string const& path)
{
	cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
	cv::Mat edges;

	cv::Canny(src, edges, 50, 150);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if(!contours.empty())
	{
		cv::Rect r = cv::boundingRect(contours.at(0));
		cv::Mat roi = src(r).clone();
		cv::imshow("Object", roi);
	}
	cv::waitKey(0);
}

void TestBlobDetection(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat dst;
	cv::cvtColor(src, dst, cv::COLOR_BGR2HSV);
	cv::cvtColor(dst, src, cv::COLOR_BGR2GRAY);

	cv::SimpleBlobDetector::Params params;
	// Thresholds
	params.minThreshold = 1.f;
	params.maxThreshold = 255.f;

	// Filter by Area
	params.filterByArea = true;
	params.minArea      = 1000.f;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity      = 0.1f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity      = 0.75f;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.00000001f;

	// Filter by Color
	params.filterByColor = false;
	params.blobColor     = 0;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	std::vector<cv::KeyPoint> keyPoints;
	detector->detect(src, keyPoints);

	std::cout << "Keypoints : " << keyPoints.size() << std::endl;

	cv::Mat imgKeypoints;
	cv::drawKeypoints(src, keyPoints, imgKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::imshow("Keypoints", imgKeypoints);
	cv::waitKey(0);
}

void TestEdgeDetection(std::string const& path)
{
	cv::Mat img = cv::imread(path);

	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	cv::Mat img_blur;
	cv::GaussianBlur(img_gray, img_blur, cv::Size(3, 3), 0);

	cv::Mat sobelx, sobely, sobelxy;
	cv::Sobel(img_blur,  sobelx, CV_64F, 1, 0, 5);
	cv::Sobel(img_blur,  sobely, CV_64F, 0, 1, 5);
	cv::Sobel(img_blur, sobelxy, CV_64F, 1, 1, 5);

	cv::imshow("Sobel X", sobelx);
	cv::imshow("Sobel Y", sobely);
	cv::imshow("Sobel XY (Sobel() Function)", sobelxy);

	cv::Mat edges;
	cv::Canny(img_blur, edges, 100, 200, 3, false);

	cv::imshow("Canny edge detection", edges);

	cv::waitKey(0);
}

void TestThresholding(std::string const& path)
{
	cv::Mat src = cv::imread(path, cv::IMREAD_GRAYSCALE);
	cv::Mat dst;

	cv::threshold(src, dst, 150, 255, cv::THRESH_BINARY);

	cv::imshow("Threshold", dst);

	cv::waitKey(0);
}

void TestColorspaces(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat dst;
	cv::cvtColor(src, dst, cv::COLOR_BGR2HSV);

	cv::imshow("Colorspace", dst);

	cv::waitKey(0);
}

// FIN FONCTIONS TESTS DE TRAITEMENT

void RunTests(std::string const& imagePath)
{
	TestBlobDetection(imagePath);
}

// MAIN : NE PAS MODIFIER
int main()
{
	try
	{
		while(true)
		{
			// Sélection de l'image à traiter
			std::vector<std::filesystem::path> files{};
			int fileChoice = -1;
			while(fileChoice == -1)
			{
				files.clear();
				for(const auto& entry : std::filesystem::directory_iterator("Assets"))
					files.emplace_back(entry.path());
				std::wcout << "Selectionnez votre image a traiter : " << std::endl;
				for(size_t i = 0; i < files.size(); ++i)
					std::cout << " " << i << " - " << files.at(i) << std::endl;
				std::cout << "-1 - Rafraichir la liste" << std::endl;
				std::cin >> fileChoice;
			}
			std::cout << "Exécution des tests..." << std::endl;

			// Exécution des tests de traitement
			RunTests(files.at(fileChoice).string());
		}
	}
	catch(EngineException const& e)
	{
		EngineException::DisplayMessageBox(e.GetCaption(), e.GetText());
	}
	catch(std::exception const& e)
	{
		EngineException::DisplayMessageBox(L"STL Exception", e.what());
	}
	catch(...)
	{
		EngineException::DisplayMessageBox(L"Unknown Exception", L"An unknown exception has occurred!");
	}

	return 0;
}
