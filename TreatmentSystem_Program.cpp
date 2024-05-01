#include "TreatmentSystem.h"

#include <opencv2/opencv.hpp>

using namespace TreatmentSystem;

void Program::Run(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	// Affichage des histogrammes
	std::set<int> colorspaceChoices;
	std::vector<std::pair<int, std::string>> colorspaces = { {0, "BGR"}, {cv::COLOR_BGR2HSV, "HSV"}, {cv::COLOR_BGR2Lab, "Lab"}, {cv::COLOR_BGR2YCrCb, "YCrCb"} };

	std::cout << "Quelles etudes d'histogramme souhaitez-vous realiser ? (-1 pour finir la selection)" << std::endl;
	for(size_t i = 0; i < colorspaces.size(); i++)
		std::cout << i << " - " << colorspaces.at(i).second << std::endl;

	int choiceIndex = -2;
	while(choiceIndex != -1)
	{
		std::cin >> choiceIndex;

		if(choiceIndex >= 0 && choiceIndex < static_cast<long long>(colorspaces.size()))
			colorspaceChoices.insert(choiceIndex);
	}

	for(auto const& choice : colorspaceChoices)
		HistogramStudy(src, colorspaces.at(choice));

	// Séparation de la feuille du fond
	std::vector<std::string> leafExtractionAlgorithms = { "Canny", "SobelXY", "Laplacian", "GrabCut" };

	std::cout << "Selectionnez l'algorithme d'extraction de la feuille : " << std::endl;
	for(size_t i = 0; i < leafExtractionAlgorithms.size(); i++)
		std::cout << i << " - " << leafExtractionAlgorithms.at(i) << std::endl;

	int leafChoiceIndex = -1;
	while(!(leafChoiceIndex >= 0 && leafChoiceIndex < static_cast<long long>(leafExtractionAlgorithms.size())))
		std::cin >> leafChoiceIndex;

	cv::Mat leaf = ExtractLeaf(src, { leafChoiceIndex, leafExtractionAlgorithms.at(leafChoiceIndex) });

}

void Program::HistogramStudy(cv::Mat const& image, std::pair<int, std::string> colorSpace)
{
	cv::Mat copy = image.clone();
	// Si l'espace de couleur sélectionné n'est pas BGR, on réalise une conversion d'espace
	if(colorSpace.first)
		cv::cvtColor(copy, copy, colorSpace.first);

	// On sépare les channels de l'image (e.g. BGR ==> B + G + R)
	std::vector<cv::Mat> color_planes;
	cv::split(copy, color_planes);

	// Initialisation des paramètres pour calculer les histogrammes de chaque channel
	int histSize = 256;
	std::array<float, 2> range = {0, 256};
	const float* histRange[] = {range.data()};
	bool bUniform = true, bAccumulate = false;

	cv::Mat a_hist, b_hist, c_hist;
	cv::calcHist(&color_planes[0], 1, 0, cv::Mat(), a_hist, 1, &histSize, histRange, bUniform, bAccumulate);
	cv::calcHist(&color_planes[1], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange, bUniform, bAccumulate);
	cv::calcHist(&color_planes[2], 1, 0, cv::Mat(), c_hist, 1, &histSize, histRange, bUniform, bAccumulate);

	int hist_width = 512, hist_height = 400;
	int bin_width = cvRound(static_cast<double>(hist_width) / histSize);
	cv::Mat histImage(hist_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));

	cv::normalize(a_hist, a_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(c_hist, c_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	// Construction de l'histogramme à partir des histogrammes des 3 channels
	for(int i = 1; i < histSize; i++)
	{
		cv::line(histImage, 
		         cv::Point(bin_width * (i - 1), hist_height - cvRound(a_hist.at<float>(i - 1))),
		         cv::Point(bin_width * i,       hist_height - cvRound(a_hist.at<float>(i))), 
		         cv::Scalar(255, 0, 0), 
		         2,
		         8, 
		         0);

		cv::line(histImage,
		         cv::Point(bin_width * (i - 1), hist_height - cvRound(b_hist.at<float>(i - 1))),
		         cv::Point(bin_width * i,       hist_height - cvRound(b_hist.at<float>(i))),
		         cv::Scalar(0, 255, 0),
		         2,
		         8,
		         0);

		cv::line(histImage,
		         cv::Point(bin_width * (i - 1), hist_height - cvRound(c_hist.at<float>(i - 1))),
		         cv::Point(bin_width * i,       hist_height - cvRound(c_hist.at<float>(i))),
		         cv::Scalar(0, 0, 255),
		         2,
		         8,
		         0);
	}

	// Affichage des différents channels + Histogramme
	cv::imshow("Colorspace - " + colorSpace.second, copy);
	cv::imshow("Colorspace - Channel A",         color_planes[0]);
	cv::imshow("Colorspace - Channel B",         color_planes[1]);
	cv::imshow("Colorspace - Channel C",         color_planes[2]);
	cv::imshow("Histogram",                      histImage);

	cv::waitKey();

	cv::destroyAllWindows();
}

cv::Mat Program::ExtractLeaf(cv::Mat const& image, std::pair<int, std::string> algorithm)
{
	cv::Mat copy = image.clone();

	cv::Mat result;

	if(algorithm.first != 3) // Tous sauf GrabCut
	{
		cv::Mat hsv;
		cv::cvtColor(copy, hsv, cv::COLOR_BGR2HSV);

		std::vector<cv::Mat> channels;
		cv::split(hsv, channels);

		cv::Mat value;
		cv::equalizeHist(channels[2], value);

		cv::Mat edges;
		switch(algorithm.first)
		{
		case 0: // Canny
			{
			cv::Canny(value, edges, 10, 123);
			}
			break;
		case 1: // SobelXY
			{
			cv::GaussianBlur(value, value, cv::Size(9, 9), 0);
			cv::Sobel(value, edges, CV_8UC1, 1, 1, 5);
			cv::threshold(edges, edges, 15, 255, cv::THRESH_BINARY);
			}
			break;
		case 2: // Laplacian
			{
			cv::GaussianBlur(value, value, cv::Size(9, 9), 0);
			cv::Laplacian(value, edges, CV_8UC1);
			cv::threshold(edges, edges, 2, 255, cv::THRESH_BINARY);
			}
			break;
		}

		cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		cv::Mat mask = edges.clone();
		cv::fillPoly(mask, contours, cv::Scalar(255, 255, 255));

		cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 12, 1, 1);
		cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

		cv::bitwise_and(copy, copy, result, mask);
	}
	else // GrabCut
	{
		cv::Mat mask;
		cv::Mat bgModel, fgModel;
		cv::grabCut(copy, mask, cv::Rect(1, 1, copy.cols - 1, copy.rows - 1), bgModel, fgModel, 25, cv::GC_INIT_WITH_RECT);

		cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);

		result = cv::Mat(copy.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		copy.copyTo(result, mask);

		cv::erode(result, result, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
	}

	cv::imshow("Result - " + algorithm.second, result);

	cv::waitKey();

	cv::destroyAllWindows();

	return result;
}

cv::Mat Program::ExtractSpots(cv::Mat const& image, std::pair<int, std::string> algorithm)
{
}
