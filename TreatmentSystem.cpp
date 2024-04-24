#include "TreatmentSystem.h"

#include <opencv2/opencv.hpp>



void TreatmentSystem::RunTests(std::string const& path)
{
	TestHSVSaturationIdea(path);
}





// ******************** TEST FUNCTIONS ************************ //

void TreatmentSystem::TestHistogram(std::string const& path)
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

void TreatmentSystem::TestHistogramGrayscale(std::string const& path)
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

void TreatmentSystem::TestHistogramHSV(std::string const& path)
{
	cv::Mat src = cv::imread(path);
	cv::cvtColor(src, src, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> bgr_planes;
	split(src, bgr_planes);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;

	cv::Mat h_hist, s_hist, v_hist;
	calcHist(&bgr_planes[0], 1, 0, cv::Mat(), h_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, cv::Mat(), s_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, cv::Mat(), v_hist, 1, &histSize, histRange, uniform, accumulate);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	normalize(h_hist, h_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(s_hist, s_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(v_hist, v_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for(int i = 1; i < histSize; i++) {
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(h_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(s_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(v_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("Source image", src);
	imshow("Source image - Hue", bgr_planes[0]);
	imshow("Source image - Saturation", bgr_planes[1]);
	imshow("Source image - Value", bgr_planes[2]);
	imshow("calcHist Demo", histImage);

	cv::imwrite("Outputs/Histogram.png", histImage);
	cv::waitKey();
}

void TreatmentSystem::TestContour(std::string const& path)
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
	cv::waitKey();
}

void TreatmentSystem::TestBlobDetection(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat copy;
	cv::medianBlur(src, copy, 21);
	//cv::cvtColor(copy, copy, cv::COLOR_BGR2HSV);
	cv::cvtColor(copy, copy, cv::COLOR_BGR2GRAY);

	cv::SimpleBlobDetector::Params params;
	// Thresholds
	params.minThreshold = 1.f;
	params.maxThreshold = 125.f;

	// Filter by Area
	params.filterByArea = true;
	params.minArea = 1000.f;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.75f;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.00000001f;

	// Filter by Color
	params.filterByColor = false;
	params.blobColor = 0;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	std::vector<cv::KeyPoint> keyPoints;
	detector->detect(copy, keyPoints);

	std::cout << "Keypoints : " << keyPoints.size() << std::endl;

	cv::Mat imgKeypoints;
	cv::drawKeypoints(src, keyPoints, imgKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::imshow("Keypoints", imgKeypoints);
	cv::waitKey();
}

void TreatmentSystem::TestEdgeDetection(std::string const& path)
{
	cv::Mat img = cv::imread(path);

	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	cv::Mat img_blur;
	cv::GaussianBlur(img_gray, img_blur, cv::Size(3, 3), 0);

	cv::Mat sobelx, sobely, sobelxy;
	cv::Sobel(img_blur, sobelx, CV_64F, 1, 0, 5);
	cv::Sobel(img_blur, sobely, CV_64F, 0, 1, 5);
	cv::Sobel(img_blur, sobelxy, CV_64F, 1, 1, 5);

	cv::imshow("Sobel X", sobelx);
	cv::imshow("Sobel Y", sobely);
	cv::imshow("Sobel XY (Sobel() Function)", sobelxy);

	cv::Mat edges;
	cv::Canny(img_blur, edges, 100, 200, 3, false);

	cv::imshow("Canny edge detection", edges);

	cv::waitKey();
}

void TreatmentSystem::TestThresholding(std::string const& path)
{
    cv::Mat src = cv::imread(path);

	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat dst;

    cv::threshold(gray, dst, 120, 255, cv::THRESH_BINARY_INV);

	cv::Mat test;
	cv::bitwise_and(src, src, test, dst);

    cv::imshow("Threshold", test);

    cv::waitKey();
}

void TreatmentSystem::TestColorspaces(std::string const& path)
{
    cv::Mat src = cv::imread(path);

    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_BGR2HSV);

	cv::namedWindow("Original");
	cv::namedWindow("Colorspace");

	auto callback = [](int action, int x, int y, int flags, void* userdata)
	{
		if(action == cv::EVENT_LBUTTONDOWN)
		{
			cv::Point2i pixel = cv::Point2i(x, y);
			cv::Scalar color = static_cast<cv::Mat*>(userdata)->at<cv::Vec3b>(x, y);

			std::cout << "Pixel aux coordonnees { X = " << pixel.x << " ; Y = " << pixel.y << " } "
			          << "= { " << color[0] << " ; " << color[1] << " ; " << color[2] << " }" << std::endl;
		}
	};

	cv::setMouseCallback("Original", callback, &src);
	cv::setMouseCallback("Colorspace", callback, &dst);
	cv::imshow("Original", src);
    cv::imshow("Colorspace", dst);

    cv::waitKey();
}

void TreatmentSystem::TestMedianBlur(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat dst;
	cv::medianBlur(src, dst, 21);

	cv::imshow("Original", src);
	cv::imshow("Median Blur", dst);

	cv::waitKey();
}

void TreatmentSystem::TestORB(std::string const& path)
{
	cv::Mat image = cv::imread(path);

	cv::Mat hsv, gray, blurred, edges;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
	cv::cvtColor(hsv, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
	cv::Canny(blurred, edges, 50, 150);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat thresholded;
	cv::threshold(gray, thresholded, 127, 255, cv::THRESH_BINARY);

	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(blurred, keypoints);

	cv::Mat imgKeypoints;
	cv::drawKeypoints(image, keypoints, imgKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::Mat labels;
	cv::Mat stats;
	cv::Mat centroids;
	cv::connectedComponentsWithStats(thresholded, labels, stats, centroids);


	cv::imshow("ORB1", image);
	cv::imshow("ORB2", hsv);
	cv::imshow("ORB3", gray);
	cv::imshow("ORB4", blurred);
	cv::imshow("ORB5", edges);
	cv::imshow("ORB6", thresholded);
	cv::imshow("ORB7", imgKeypoints);
	cv::imshow("ORB8", labels);
	cv::imshow("ORB9", stats);
	cv::imshow("ORB10", centroids);

	cv::waitKey();
}

void TreatmentSystem::TestSegmentation(std::string const& path)
{
	cv::Mat img = cv::imread(path);

	cv::Mat hsv;
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask1, mask2;
	cv::inRange(hsv, cv::Scalar(0, 100, 20), cv::Scalar(10, 255, 255), mask1);
	cv::inRange(hsv, cv::Scalar(170, 100, 20), cv::Scalar(180, 255, 255), mask2);

	cv::Mat mask = mask1 | mask2;

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat res = img.clone();
	for(int i = 0; i < contours.size(); ++i)
	{
		cv::drawContours(res, contours, i, cv::Scalar(0, 255, 0));

		cv::RotatedRect r = cv::minAreaRect(contours.at(i));
		cv::Point2f pts[4];
		r.points(pts);

		for(int j = 0; j < 4; ++j)
		{
			cv::line(res, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 255));
		}

		cv::Rect box = cv::boundingRect(contours[i]);
		cv::rectangle(res, box, cv::Scalar(255, 0, 0));
	}

	cv::imshow("Original", img);
	cv::imshow("Segmented", res);

	cv::waitKey();
}

void TreatmentSystem::TestHSVSaturationIdea(std::string const& path)
{
	cv::Mat src = cv::imread(path);
	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_planes;
	split(hsv, hsv_planes);

	cv::Mat sat = hsv_planes[1];


	cv::Mat blurred;
	cv::medianBlur(sat, blurred, 7);


	cv::SimpleBlobDetector::Params params;
	// Thresholds
	params.minThreshold = 1.f;
	params.maxThreshold = 70.f;

	// Filter by Area
	params.filterByArea = true;
	params.minArea = 100.f;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.75f;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.00000001f;

	// Filter by Color
	params.filterByColor = false;
	params.blobColor = 0;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	std::vector<cv::KeyPoint> keypoints;
	detector->detect(blurred, keypoints);

	std::cout << "Keypoints : " << keypoints.size() << std::endl;

	cv::Mat imgKeypoints;
	cv::drawKeypoints(src, keypoints, imgKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::imshow("Original", src);
	cv::imshow("HSV", hsv);
	cv::imshow("Saturation", sat);
	cv::imshow("Blur", blurred);
	cv::imshow("Keypoints", imgKeypoints);
	cv::waitKey();

}

// ****************** END TEST FUNCTIONS ********************** //
