#include <EngineException.h>
#include <Image.h>
#include <Window.h>

int main()
{
	try
	{
		Image  image("Assets/Test.jpg");
		Window window("Display Window");

		image.SetRGB(400, 400, { 255, 0, 0 });

		window.DisplayImage(image);
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
