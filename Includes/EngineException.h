#pragma once
#include <Windows.h>
#include <exception>
#include <string>

#define EXCEPTION(title, description) EngineException(title, description, __FILE__, __LINE__)

class EngineException : public std::exception
{
public:
    EngineException(std::wstring title, std::wstring description, std::string file, unsigned int line)
	: title(std::move(title))
    , description(std::move(description))
    , file(std::move(file))
    , line(line)
    {}

    static void DisplayMessageBox(std::wstring const& caption, const char* text)
    {
        std::string        toString     = std::string(text);
        std::wstring const toWideString = std::wstring(toString.begin(), toString.end());
        DisplayMessageBox(caption, toWideString);
    }

    static void DisplayMessageBox(std::wstring const& caption, std::wstring const& text)
    {
        MessageBox(nullptr, text.c_str(), caption.c_str(), MB_ICONERROR | MB_OK);
    }

    std::wstring GetText() const
    {

        return L"[File] : "            + std::wstring(file.begin(), file.end())
    	     + L"\n[Line] : "          + std::to_wstring(line)
    	     + L"\n[Description] : \n" + description;
    }

    std::wstring GetCaption() const
    {
        return title;
    }

private:
    std::wstring title;
    std::wstring description;
    std::string file;
    unsigned int line;
};
