#include "TreatmentSystem.h"

#include <opencv2/opencv.hpp>

using namespace TreatmentSystem;

void Test::RunTests(std::string const& path)
{
	TestGrabCut(path);
}


// ******************** TEST FUNCTIONS ************************ //

void Test::TestHistogram(std::string const& path)
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

void Test::TestHistogramGrayscale(std::string const& path)
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

void Test::TestHistogramHSV(std::string const& path)
{
	cv::Mat src = cv::imread(path);
	cv::cvtColor(src, src, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_planes;
	split(src, hsv_planes);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;

	cv::Mat h_hist, s_hist, v_hist;
	calcHist(&hsv_planes[0], 1, 0, cv::Mat(), h_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&hsv_planes[1], 1, 0, cv::Mat(), s_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&hsv_planes[2], 1, 0, cv::Mat(), v_hist, 1, &histSize, histRange, uniform, accumulate);

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
	imshow("Source image - Hue", hsv_planes[0]);
	imshow("Source image - Saturation", hsv_planes[1]);
	imshow("Source image - Value", hsv_planes[2]);
	imshow("calcHist Demo", histImage);

	cv::imwrite("Outputs/Histogram.png", histImage);
	cv::waitKey();
}

void Test::TestHistogramLab(std::string const& path)
{
	cv::Mat src = cv::imread(path);
	cv::cvtColor(src, src, cv::COLOR_BGR2Lab);

	std::vector<cv::Mat> lab_planes;
	split(src, lab_planes);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;

	cv::Mat l_hist, a_hist, b_hist;
	calcHist(&lab_planes[0], 1, 0, cv::Mat(), l_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&lab_planes[1], 1, 0, cv::Mat(), a_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&lab_planes[2], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	normalize(l_hist, l_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(a_hist, a_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for(int i = 1; i < histSize; i++) {
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(l_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(l_hist.at<float>(i))),
			cv::Scalar(255, 255, 255), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(a_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(a_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}

	imshow("Source image", src);
	imshow("Source image - Lightness", lab_planes[0]);
	imshow("Source image - A*", lab_planes[1]);
	imshow("Source image - B*", lab_planes[2]);
	imshow("calcHist Demo", histImage);

	cv::imwrite("Outputs/Histogram.png", histImage);
	cv::waitKey();
}

void Test::TestContour(std::string const& path)
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

void Test::TestBlobDetection(std::string const& path)
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

void Test::TestEdgeDetection(std::string const& path)
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

void Test::TestThresholding(std::string const& path)
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

void Test::TestColorspaces(std::string const& path)
{
    cv::Mat src = cv::imread(path);

    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);

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

void Test::TestMedianBlur(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat dst;
	cv::medianBlur(src, dst, 21);

	cv::imshow("Original", src);
	cv::imshow("Median Blur", dst);

	cv::waitKey();
}

void Test::TestORB(std::string const& path)
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

void Test::TestSegmentation(std::string const& path)
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

void Test::TestHSVSaturationIdea(std::string const& path)
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

void Test::TestLabDarkenIdea(std::string const& path)
{
	cv::Mat src = cv::imread(path);
	cv::Mat lab;

	cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

	std::vector<cv::Mat> lab_planes;
	split(lab, lab_planes);

	cv::Mat b_star = lab_planes[2];

	cv::Mat blurred;
	cv::medianBlur(b_star, blurred, 7);


	cv::SimpleBlobDetector::Params params;
	// Thresholds
	params.minThreshold = 1.f;
	params.maxThreshold = 176.f;

	// Filter by Area
	params.filterByArea = true;
	params.minArea = 100.f;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.01f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.5f;

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
	cv::imshow("Lab", lab);
	cv::imshow("B*", b_star);
	cv::imshow("Blur", blurred);
	cv::imshow("Keypoints", imgKeypoints);

	cv::waitKey();
}

void Test::TestContourDetection(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	cv::Mat gray;
	cv::cvtColor(hsv, gray, cv::COLOR_BGR2GRAY);

	cv::Mat mask;
	cv::inRange(gray, cv::Scalar(0, 0, 0), cv::Scalar(100, 100, 100), mask);

	cv::Mat laplacianImage;
	cv::Laplacian(gray, laplacianImage, CV_64F);

	cv::Scalar mean, stddev;
	cv::meanStdDev(laplacianImage, mean, stddev, cv::Mat());
	double variance = stddev.val[0] * stddev.val[0];
	if(variance <= 10)
	{
		std::cout << "L'image est floue" << std::endl;
		return;
	}

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for(auto const& contour : contours)
	{
		cv::Rect boundingRect = cv::boundingRect(contour);
		cv::rectangle(src, boundingRect, cv::Scalar(0, 0, 255), 2);
	}

	cv::imshow("Resultat", src);

	cv::waitKey();
}

void Test::TestBackgroundSubtractor(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();

	cv::Mat fgMask;
	pBackSub->apply(src, fgMask);

	cv::imshow("Original", src);
	cv::imshow("Mask", fgMask);

	cv::waitKey();
}

void Test::TestHistogramEqualization(std::string const& path)
{
	cv::Mat src = cv::imread(path);


	cv::Mat equalizedImage;
	cv::cvtColor(src, equalizedImage, cv::COLOR_BGR2YCrCb);

	std::vector<cv::Mat> vec_channels;
	cv::split(equalizedImage, vec_channels);

	cv::equalizeHist(vec_channels[0], vec_channels[0]);

	cv::merge(vec_channels, equalizedImage);

	cv::cvtColor(equalizedImage, equalizedImage, cv::COLOR_YCrCb2BGR);

	cv::cvtColor(equalizedImage, equalizedImage, cv::COLOR_BGR2GRAY);

	cv::medianBlur(equalizedImage, equalizedImage, 7);

	cv::Mat thresh;

	cv::adaptiveThreshold(equalizedImage, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 9, -1);

	cv::medianBlur(thresh, thresh, 3);
	cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 4, 1, 1);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat mask = thresh.clone();
	//cv::drawContours(srcContours, contours, -1, cv::Scalar(255, 0, 0), 2); 
	cv::fillPoly(mask, contours, cv::Scalar(255, 255, 255));

	cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 40, 1, 1);
	cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 30, 1, 1);

	cv::Mat cutout;
	cv::bitwise_and(src, src, cutout, mask);

	cv::cvtColor(cutout, cutout, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> channels;
	cv::split(cutout, channels);

	for(int i = 0; i < channels[1].rows; i++)
		for(int j = 0; j < channels[1].cols; j++)
			if(channels[1].at<cv::Vec3b>(i, j) == cv::Vec3b{0, 0, 0})
				channels[1].at<cv::Vec3b>(i, j) = cv::Vec3b{ 255, 255, 255 };

	cv::Mat sat = channels[1].clone();

	cv::medianBlur(sat, sat, 7);

	//cv::equalizeHist(sat, sat);

	cv::Mat invSat;
	cv::bitwise_not(sat, invSat);





	cv::Mat threshedInvSat;
	cv::threshold(invSat, threshedInvSat, 199, 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point>> contoursss;
	cv::findContours(threshedInvSat, contoursss, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat testContours = src.clone();

	for(int i = 0; i < contoursss.size(); i++)
	{
		// Affichage des contours en bleu
		cv::drawContours(testContours, contoursss, i, cv::Scalar(255, 0, 0));

		// Affichage d'un rectangle avec rotation englobant le contour
		cv::RotatedRect r = cv::minAreaRect(contoursss[i]);
		cv::Point2f pts[4];
		r.points(pts);
		for(int j = 0; j < 4; j++)
			cv::line(testContours, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 255));

		// Affichage boîte englobante
		cv::Rect box = cv::boundingRect(contoursss[i]);
		cv::rectangle(testContours, box, cv::Scalar(0, 255, 0));
	}

	/*cv::SimpleBlobDetector::Params params;
	// Thresholds
	params.minThreshold = 1.f;
	params.maxThreshold = 102.f;

	// Filter by Area
	params.filterByArea = true;
	params.minArea = 200.f;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.01f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.65f;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.1f;

	// Filter by Color
	params.filterByColor = false;
	params.blobColor = 0;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	std::vector<cv::KeyPoint> keypoints;
	detector->detect(sat, keypoints);

	std::cout << "Keypoints : " << keypoints.size() << std::endl;

	cv::Mat imgKeypoints = src.clone();

	for(auto keypoint : keypoints)
	{
		cv::Rect rec = cv::Rect(keypoint.pt.x - keypoint.size / 2, keypoint.pt.y - keypoint.size / 2, keypoint.size, keypoint.size);
		cv::rectangle(imgKeypoints, rec, cv::Scalar(0, 0, 255), 2);
	}*/

	cv::imshow("Original", src);
	cv::imshow("Equalized", equalizedImage);
	cv::imshow("Threshold", thresh);
	cv::imshow("Contours", mask);
	cv::imshow("Cutout", cutout);
	cv::imshow("Equalized Saturation", threshedInvSat);
	cv::imshow("Test contours", testContours);
	//cv::imshow("Keypoints", imgKeypoints);

	cv::waitKey();

	cv::destroyAllWindows();
}

void Test::TestLeafCanny(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_channels;
	cv::split(hsv, hsv_channels);

	cv::Mat value;

	cv::equalizeHist(hsv_channels[2], value);
	//cv::equalizeHist(hsv_channels[1], value);

	cv::Mat edges;
	cv::Canny(value, edges, 10, 123);

	cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat mask = edges.clone();
	cv::fillPoly(mask, contours, cv::Scalar(255, 255, 255));

	cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 12, 1, 1);
	cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

	cv::Mat cutout;
	cv::bitwise_and(src, src, cutout, mask);

	cv::imshow("Original", src);
	cv::imshow("Valeur égalisé", value);
	cv::imshow("Edges", edges);
	cv::imshow("Mask", mask);
	cv::imshow("Cutout", cutout);

	cv::imwrite("Outputs/Canny.png", cutout);

	cv::waitKey();
}

void Test::TestLeafSobel(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_channels;
	cv::split(hsv, hsv_channels);

	cv::Mat value = hsv_channels[2];

	cv::equalizeHist(hsv_channels[2], value);
	cv::GaussianBlur(value, value, cv::Size(9, 9), 0);

	cv::Mat edges;
	cv::Sobel(value, edges, CV_8UC1, 1, 1, 5);

	cv::threshold(edges, edges, 15, 255, cv::THRESH_BINARY);

	cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat mask = edges.clone();
	cv::fillPoly(mask, contours, cv::Scalar(255, 255, 255));

	cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 12, 1, 1);
	cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

	cv::Mat cutout;
	cv::bitwise_and(src, src, cutout, mask);

	cv::imshow("Original", src);
	cv::imshow("Valeur égalisé", value);
	cv::imshow("Edges", edges);
	cv::imshow("Mask", mask);
	cv::imshow("Cutout", cutout);

	cv::imwrite("Outputs/Sobel.png", cutout);

	cv::waitKey();
}

void Test::TestLeafLaplacian(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_channels;
	cv::split(hsv, hsv_channels);

	cv::Mat value = hsv_channels[2];

	cv::equalizeHist(hsv_channels[2], value);
	cv::GaussianBlur(value, value, cv::Size(9, 9), 0);

	cv::Mat edges;
	cv::Laplacian(value, edges, CV_8UC1);

	
	cv::threshold(edges, edges, 2, 255, cv::THRESH_BINARY);

	cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat mask = edges.clone();
	cv::fillPoly(mask, contours, cv::Scalar(255, 255, 255));

	cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 12, 1, 1);
	cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

	
	cv::Mat cutout;
	cv::bitwise_and(src, src, cutout, mask);
	
	cv::imshow("Original", src);
	cv::imshow("Valeur égalisé", value);
	cv::imshow("Edges", edges);
	cv::imshow("Mask", mask);
	cv::imshow("Cutout", cutout);

	cv::imwrite("Outputs/Laplacian.png", cutout);

	cv::waitKey();
}

void Test::TestGrabCut(std::string const& path)
{
	cv::Mat src     = cv::imread(path);

	cv::Mat grabCutSrc = src.clone();

	cv::Mat result;
	cv::Mat bgModel, fgModel;

	cv::grabCut(grabCutSrc, result, cv::Rect(1, 1, src.cols - 1, src.rows - 1), bgModel, fgModel, 25, cv::GC_INIT_WITH_RECT);

	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);

	cv::Mat foreground(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	grabCutSrc.copyTo(foreground, result);

	cv::erode(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);

	/*
	std::vector<cv::Mat> bgr_channels;
	cv::split(foreground, bgr_channels);

	cv::Mat green = bgr_channels[1];

	cv::SimpleBlobDetector::Params params;
	// Thresholds
	params.minThreshold = 80.f;
	params.maxThreshold = 111.f;

	// Filter by Area
	params.filterByArea = true;
	params.minArea = 1000.f;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1f;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.5f;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.00000001f;

	// Filter by Color
	params.filterByColor = false;
	params.blobColor = 0;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	std::vector<cv::KeyPoint> keyPoints;
	detector->detect(green, keyPoints);

	std::cout << "Keypoints : " << keyPoints.size() << std::endl;

	cv::Mat imgKeypoints;
	cv::drawKeypoints(foreground, keyPoints, imgKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::imshow("Keypoints", imgKeypoints);
	*/

	cv::Mat lab;
	cv::cvtColor(foreground, lab, cv::COLOR_BGR2Lab);

	std::vector<cv::Mat> lab_channels;
	cv::split(lab, lab_channels);

	cv::Mat mask;
	cv::threshold(lab_channels[1], mask, 115, 255, cv::THRESH_BINARY);

	cv::Mat cutout;
	cv::bitwise_and(foreground, foreground, cutout, mask);

	cv::erode(cutout, cutout, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);

	cv::cvtColor(cutout, cutout, cv::COLOR_BGR2GRAY);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(cutout, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	for(int i = 0; i < contours.size(); i++)
	{
		// Affichage des contours en bleu
			cv::drawContours(src, contours, i, cv::Scalar(255, 0, 0));

		// Affichage d'un rectangle avec rotation englobant le contour
			cv::RotatedRect r = cv::minAreaRect(contours[i]);
		cv::Point2f pts[4];
		r.points(pts);
		for(int j = 0; j < 4; j++)
			cv::line(src, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 255));

		// Affichage boîte englobante
		cv::Rect box = cv::boundingRect(contours[i]);
		cv::rectangle(src, box, cv::Scalar(0, 255, 0));

		std::cout << "\n----------------------------------"
		          << "\nChemin image : " << path
		          << "\nID zone : " << i + 1
		          << "\nCoordonnees : ( X = " << box.x << " ; Y = " << box.y << " )"
		          << "\nLargeur : " << box.width
		          << "\nHauteur : " << box.height
		          << "\n----------------------------------" << std::endl;
	}

	/*cv::Mat hsv;
	cv::cvtColor(foreground, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_channels;
	cv::split(hsv, hsv_channels);

	for(int i = 0; i < hsv_channels[1].rows; i++)
		for(int j = 0; j < hsv_channels[1].cols; j++)
			if(hsv_channels[1].at<uchar>(i, j) == 0)
				hsv_channels[1].at<uchar>(i, j) = 255;

	cv::Mat saturation_equalized;
	cv::equalizeHist(hsv_channels[1], saturation_equalized);

	cv::bitwise_not(saturation_equalized, saturation_equalized);
	cv::medianBlur(saturation_equalized, saturation_equalized, 9);
	cv::threshold(saturation_equalized, saturation_equalized, 230, 255, cv::THRESH_BINARY);

	cv::erode(saturation_equalized, saturation_equalized, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);
	cv::dilate(saturation_equalized, saturation_equalized, cv::Mat(), cv::Point(-1, -1), 9, 1, 1);

	cv::Mat cutout;
	cv::bitwise_and(foreground, foreground, cutout, saturation_equalized);

	cv::Mat cutout_lab;
	cv::cvtColor(cutout, cutout_lab, cv::COLOR_BGR2Lab);

	std::vector<cv::Mat> lab_channels;
	cv::split(cutout_lab, lab_channels);

	for(int i = 0; i < lab_channels[0].rows; i++)
		for(int j = 0; j < lab_channels[0].cols; j++)
			if(lab_channels[0].at<uchar>(i, j) == 0)
				lab_channels[0].at<uchar>(i, j) = 255;

	cv::GaussianBlur(lab_channels[0], lab_channels[0], cv::Size(5, 5), 0);
	cv::Mat invertedL;
	cv::bitwise_not(lab_channels[0], invertedL);
	cv::equalizeHist(invertedL, invertedL);

	cv::Mat edges;
	cv::Canny(invertedL, edges, 150, 250);
	
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(saturation_equalized, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	for(int i = 0; i < contours.size(); i++)
	{
		 Affichage des contours en bleu
		cv::drawContours(foreground, contours, i, cv::Scalar(255, 0, 0));

		 Affichage d'un rectangle avec rotation englobant le contour
		cv::RotatedRect r = cv::minAreaRect(contours[i]);
		cv::Point2f pts[4];
		r.points(pts);
		for(int j = 0; j < 4; j++)
			cv::line(foreground, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 255));

		 Affichage boîte englobante
		cv::Rect box = cv::boundingRect(contours[i]);
		cv::rectangle(foreground, box, cv::Scalar(0, 255, 0));
	}*/

	//cv::threshold(lab_channels[1], a_star, 100, 255, cv::THRESH_BINARY);
	
	cv::imshow("Original", src);
	cv::imshow("GrabCut", foreground);
	cv::imshow("Grabcut Mask", mask);
	cv::imshow("Grabcut Cutout", cutout);
	//cv::imshow("GrabCut Mask", saturation_equalized);
	//cv::imshow("GrabCut Cutout", cutout);
	//cv::imshow("GrabCut Cutout - L inverted", invertedL);
	//cv::imshow("GrabCut Edges", edges);
	//cv::imshow("GrabCut Edges", edges);

	cv::imwrite("Outputs/AStarTâches.png", src);

	cv::waitKey();
}

void Test::TestNegative(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	std::vector<cv::Mat> bgr_channels;
	cv::split(src, bgr_channels);

	cv::bitwise_not(bgr_channels[0], bgr_channels[0]);
	cv::equalizeHist(bgr_channels[0], bgr_channels[0]);
	cv::bitwise_not(bgr_channels[1], bgr_channels[1]);
	//cv::equalizeHist(bgr_channels[1], bgr_channels[1]);
	cv::bitwise_not(bgr_channels[2], bgr_channels[2]);
	//cv::equalizeHist(bgr_channels[2], bgr_channels[2]);

	cv::merge(bgr_channels, src);

	cv::imshow("Negative", src);

	cv::waitKey();
}

void Test::DisplayColorspaces(std::string const& path)
{
	cv::Mat bgr = cv::imread(path);

	std::vector<cv::Mat> bgr_channels;
	cv::split(bgr, bgr_channels);

	cv::equalizeHist(bgr_channels[0], bgr_channels[0]);
	cv::equalizeHist(bgr_channels[1], bgr_channels[1]);
	cv::equalizeHist(bgr_channels[2], bgr_channels[2]);

	cv::Mat hsv;
	cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_channels;
	cv::split(hsv, hsv_channels);

	cv::equalizeHist(hsv_channels[0], hsv_channels[0]);
	cv::equalizeHist(hsv_channels[1], hsv_channels[1]);
	cv::equalizeHist(hsv_channels[2], hsv_channels[2]);

	cv::Mat lab;
	cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

	std::vector<cv::Mat> lab_channels;
	cv::split(lab, lab_channels);

	cv::equalizeHist(lab_channels[0], lab_channels[0]);
	cv::equalizeHist(lab_channels[1], lab_channels[1]);
	cv::equalizeHist(lab_channels[2], lab_channels[2]);

	cv::Mat yuv;
	cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YCrCb);

	std::vector<cv::Mat> yuv_channels;
	cv::split(yuv, yuv_channels);

	cv::equalizeHist(yuv_channels[0], yuv_channels[0]);
	cv::equalizeHist(yuv_channels[1], yuv_channels[1]);
	cv::equalizeHist(yuv_channels[2], yuv_channels[2]);

	cv::Mat gray;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

	cv::imshow("RGB",             bgr);
	cv::imshow("Red",             bgr_channels[2]);
	cv::imshow("Green",           bgr_channels[1]);
	cv::imshow("Blue",            bgr_channels[0]);
	cv::imshow("HSV",             hsv);
	cv::imshow("Hue",             hsv_channels[0]);
	cv::imshow("Saturation",      hsv_channels[1]);
	cv::imshow("Value",           hsv_channels[2]);
	cv::imshow("L*a*b*",          lab);
	cv::imshow("L*",              lab_channels[0]);
	cv::imshow("a*",              lab_channels[1]);
	cv::imshow("b*",              lab_channels[2]);
	cv::imshow("YCrCb",           yuv);
	cv::imshow("Luma",            yuv_channels[0]);
	cv::imshow("Red-difference",  yuv_channels[1]);
	cv::imshow("Blue-difference", yuv_channels[2]);
	cv::imshow("Grayscale",       gray);

	cv::imwrite("Outputs/Red.png", bgr_channels[2]);
	cv::imwrite("Outputs/Green.png", bgr_channels[1]);
	cv::imwrite("Outputs/Blue.png", bgr_channels[0]);
	cv::imwrite("Outputs/HSV.png", hsv);
	cv::imwrite("Outputs/Hue.png", hsv_channels[0]);
	cv::imwrite("Outputs/Saturation.png", hsv_channels[1]);
	cv::imwrite("Outputs/Value.png", hsv_channels[2]);
	cv::imwrite("Outputs/Lab.png", lab);
	cv::imwrite("Outputs/L.png", lab_channels[0]);
	cv::imwrite("Outputs/a.png", lab_channels[1]);
	cv::imwrite("Outputs/b.png", lab_channels[2]);
	cv::imwrite("Outputs/YCrCb.png", yuv);
	cv::imwrite("Outputs/Y.png", yuv_channels[0]);
	cv::imwrite("Outputs/Cr.png", yuv_channels[1]);
	cv::imwrite("Outputs/Cb.png", yuv_channels[2]);
	cv::imwrite("Outputs/Grayscale.png", gray);

	cv::waitKey();
}

// ****************** END TEST FUNCTIONS ********************** //
