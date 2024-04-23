#include <EngineException.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

#include "../TreatmentSystem.h"

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
			TreatmentSystem::GetInstance().RunTests(files.at(fileChoice).string());
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
