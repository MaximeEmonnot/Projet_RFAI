#include <EngineException.h>
#include <Image.h>
#include <Window.h>

#include "Transformations.h"

int main()
{
	try
	{
		Image  image("Assets/Test.jpg");
		Window window("Display Window");

		Image grayscale = Transformations::ToGrayScale(image);

		window.DisplayImage(grayscale);
	}
	catch(EngineException const& e)
	{
		e.DisplayMessageBox();
	}
	catch(std::exception const& e)
	{
		
	}
	catch(...)
	{
		
	}

	return 0;
}
