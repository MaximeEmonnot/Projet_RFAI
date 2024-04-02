#include <EngineException.h>
#include <Image.h>
#include <Window.h>

int main()
{
	try
	{
		Image  const image("Assets/Test.jpg");
		Window const window("Display Window");

		window.DisplayImage(image);
	}
	catch(EngineException const& e)
	{
		e.DisplayMessageBox();
	}

	return 0;
}
