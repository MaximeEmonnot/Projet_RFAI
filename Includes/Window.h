#pragma once
#include <string>

#include <Image.h>

class Window
{
public:
    Window(std::string const& name);
    ~Window();
    Window(Window const&)            = default;
    Window& operator=(Window const&) = default;
    Window(Window&&)                 = default;
    Window& operator=(Window&&)      = default;

    void DisplayImage(Image const& image) const;

private:
    std::string name;
};
