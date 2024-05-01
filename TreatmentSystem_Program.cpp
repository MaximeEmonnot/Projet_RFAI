#include "TreatmentSystem.h"

#include <opencv2/opencv.hpp>

using namespace TreatmentSystem;

void Program::Run(std::string const& path)
{
	cv::Mat src = cv::imread(path);

	// Affichage des histogrammes
	std::set<int> colorspaceChoices;
	std::vector<std::pair<int, std::string>> colorspaces = { {0, "BGR"}, {cv::COLOR_BGR2HSV, "HSV"}, {cv::COLOR_BGR2Lab, "Lab"}, {cv::COLOR_BGR2YCrCb, "YCrCb"} };

	std::cout << "Quels espaces de couleur souhaitez vous etudier ? (-1 pour finaliser la selection)" << std::endl;
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

	// Recherche des tâches brunes
	std::vector<std::string> spotSearchAlgorithms = { "Detection de contours (Canny)", "Detection de contours (SobelXY)", "Detection de contours (Laplacien)", "Masque a*", "SimpleBlobDetector"};

	std::cout << "Selectionnez l'algorithme de recherche de taches : " << std::endl;
	for(size_t i = 0; i < spotSearchAlgorithms.size(); i++)
		std::cout << i << " - " << spotSearchAlgorithms.at(i) << std::endl;

	int spotChoiceIndex = -1;
	while(!(spotChoiceIndex >= 0 && spotChoiceIndex < static_cast<long long>(spotSearchAlgorithms.size())))
		std::cin >> spotChoiceIndex;

	cv::Mat result = ExtractSpots(src, leaf, { spotChoiceIndex, spotSearchAlgorithms.at(spotChoiceIndex) });

	// Enregistrement des résultats
	char answer{};
	std::cout << "Souhaitez-vous enregistrer le resultat ? (o/n)" << std::endl;
	std::cin >> answer;
	
	if(answer == 'o' || answer == 'O' || answer == 'y' || answer == 'Y')
	{
		std::string filePath;
		std::cout << "Entrez le nom du fichier que vous souhaitez enregistrer : " << std::endl;
		std::cin >> filePath;

		std::string const finalPath = "Outputs/" + filePath + ".png";
		std::cout << "Enregistrement des resultats dans : " << finalPath << std::endl;
		cv::imwrite(finalPath, result);
	}
}

void Program::HistogramStudy(cv::Mat const& image, std::pair<int, std::string> const& colorSpace)
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

cv::Mat Program::ExtractLeaf(cv::Mat const& image, std::pair<int, std::string> const& algorithm)
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
		default: break;
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

cv::Mat Program::ExtractSpots(cv::Mat const& image, cv::Mat const& leaf, std::pair<int, std::string> const& algorithm)
{
	cv::Mat result   = image.clone();
	cv::Mat copyLeaf = leaf.clone();

	if(algorithm.first < 3) // Détection de contour (Canny, SobelXY, Laplacien)
	{
		// Prétraitement : le fond noir devient blanc
		for(int x = 0; x < copyLeaf.rows; x++)
			for(int y = 0; y < copyLeaf.cols; y++)
				if(copyLeaf.at<cv::Vec3b>(x, y) == cv::Vec3b{0, 0, 0})
					copyLeaf.at<cv::Vec3b>(x, y) = cv::Vec3b{ 255, 255, 255 };

		std::vector<cv::Mat> channels;
		cv::split(copyLeaf, channels);

		cv::Mat green;
		cv::equalizeHist(channels[1], green);
		cv::bitwise_not(green, green);

		cv::threshold(green, green, 252, 255, cv::THRESH_BINARY);

		cv::erode(green, green, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
		cv::dilate(green, green, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);

		cv::Mat edges;
		switch(algorithm.first)
		{
		case 0: // Canny
		{
			cv::Canny(green, edges, 144, 192);
		}
		break;
		case 1: // SobelXY
		{
			cv::GaussianBlur(green, green, cv::Size(9, 9), 0);
			cv::Sobel(green, edges, CV_8UC1, 1, 1, 5);
			cv::threshold(edges, edges, 15, 255, cv::THRESH_BINARY);
		}
		break;
		case 2: // Laplacian
		{
			cv::GaussianBlur(green, green, cv::Size(9, 9), 0);
			cv::Laplacian(green, edges, CV_8UC1);
			cv::threshold(edges, edges, 2, 255, cv::THRESH_BINARY);
		}
		break;
		default: break;
		}

		cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		for(size_t i = 0; i < contours.size(); i++)
		{
			// Affichage des contours en bleu
			cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(255, 0, 0));

			// Affichage d'un rectangle avec rotation englobant le contour
			cv::RotatedRect r = cv::minAreaRect(contours.at(i));
			cv::Point2f pts[4];
			r.points(pts);
			for(int j = 0; j < 4; j++)
				cv::line(result, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 255));

			// Affichage boîte englobante
			cv::Rect box = cv::boundingRect(contours[i]);
			cv::rectangle(result, box, cv::Scalar(0, 255, 0));

			std::cout << "\n----------------------------------"
				<< "\nID zone : " << i + 1
				<< "\nCoordonnees : ( X = " << box.x << " ; Y = " << box.y << " )"
				<< "\nLargeur : " << box.width
				<< "\nHauteur : " << box.height
				<< "\n----------------------------------" << std::endl;
		}
	}
	else if(algorithm.first == 3) // Masque à l'aide du channel a* (espace de couleur L*a*b*)
	{
		cv::Mat lab;
		cv::cvtColor(copyLeaf, lab, cv::COLOR_BGR2Lab);

		std::vector<cv::Mat> channels;
		cv::split(lab, channels);

		cv::Mat mask;
		cv::threshold(channels[1], mask, 115, 255, cv::THRESH_BINARY);

		cv::Mat cutout;
		cv::bitwise_and(copyLeaf, copyLeaf, cutout, mask);

		cv::erode(cutout, cutout, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
		cv::dilate(cutout, cutout, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

		cv::cvtColor(cutout, cutout, cv::COLOR_BGR2GRAY);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(cutout, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		for(size_t i = 0; i < contours.size(); i++)
		{
			// Affichage des contours en bleu
			cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(255, 0, 0));

			// Affichage d'un rectangle avec rotation englobant le contour
			cv::RotatedRect r = cv::minAreaRect(contours.at(i));
			cv::Point2f pts[4];
			r.points(pts);
			for(int j = 0; j < 4; j++)
				cv::line(result, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 0, 255));

			// Affichage boîte englobante
			cv::Rect box = cv::boundingRect(contours[i]);
			cv::rectangle(result, box, cv::Scalar(0, 255, 0));

			std::cout << "\n----------------------------------"
				<< "\nID zone : " << i + 1
				<< "\nCoordonnees : ( X = " << box.x << " ; Y = " << box.y << " )"
				<< "\nLargeur : " << box.width
				<< "\nHauteur : " << box.height
				<< "\n----------------------------------" << std::endl;
		}
	}
	else // SimpleBlobDetector
	{
		std::vector<cv::Mat> channels;
		cv::split(copyLeaf, channels);

		cv::Mat green = channels[1];

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
		params.minInertiaRatio = 0.01f;

		// Filter by Color
		params.filterByColor = false;
		params.blobColor = 0;

		cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

		std::vector<cv::KeyPoint> keyPoints;
		detector->detect(green, keyPoints);

		cv::drawKeypoints(result, keyPoints, result, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		/*
		 * Affichage des coordonnées des blobs
		 * Bien que ce ne soient pas des rectangles mais des cercles et que nous attendons surtout des rectangles englobant les tâches,
		 * le résultat est néanmoins intéressant pour la reconnaissance des zones de pixels ayant des caractéristiques proches.
		 */
		for(size_t i = 0; i < keyPoints.size(); i++)
		{
			cv::KeyPoint const keyPoint = keyPoints.at(i);

			std::cout << "\n----------------------------------"
				<< "\nID zone : " << i + 1
				<< "\nCoordonnees : ( X = " << keyPoint.pt.x << " ; Y = " << keyPoint.pt.y << " )"
				<< "\nRayon zone : " << keyPoint.size / 2
				<< "\n----------------------------------" << std::endl;
		}
	}

	cv::imshow("Result - " + algorithm.second, result);

	cv::waitKey();

	cv::destroyAllWindows();

	return result;
}
