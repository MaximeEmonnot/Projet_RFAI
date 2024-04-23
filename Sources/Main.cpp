#include <EngineException.h>
#include <Image.h>
#include <Transformations.h>
#include <Window.h>

void Test()
{
	
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
	// Test();
	Safe();

	return 0;
}
