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
	//std::cout << "Selectionnez l'algorithme d'extraction de la feuille : " << std::endl;
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

cv::Mat Program::ExtractLeaf(cv::Mat const& image)
{
}

cv::Mat Program::ExtractSpots(cv::Mat const& image)
{
}
