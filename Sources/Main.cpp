#include <EngineException.h>
#include <Image.h>
#include <Transformations.h>
#include <Window.h>

void Test()
{
    cv::Mat src = cv::imread(cv::samples::findFile("Assets / IMG_0225.jpg"), cv::IMREAD_COLOR);

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
	cv::waitKey();
}

void Safe()
{
	try
	{
		Image  const image("Assets/Test.jpg");
		Window const window("Original Window");

		Image  const filter = Transformations::ToGrayScale(image);
		Window const filterWindow("Filter Window");

		window.DisplayImage(image);
		filterWindow.DisplayImage(filter);
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
}

int main()
{
	// Commenter l'un ou l'autre
	// Test : Tous les bouts de code qu'on souhaiterais tester à part, sans se poser de question concernant une bonne structure ou non
	// Safe : Code bien structuré
	Test();
	// Safe();

	return 0;
}
