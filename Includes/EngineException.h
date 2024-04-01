#pragma once
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

    void DisplayMessageBox() const
	{
        MessageBox(nullptr, GetText().c_str(), GetTitle().c_str(), MB_ICONERROR | MB_OK);
    }

    std::wstring GetText() const
    {

        return L"[File] : "            + std::wstring(file.begin(), file.end())
    	     + L"\n[Line] : "          + std::to_wstring(line)
    	     + L"\n[Description] : \n" + description;
    }

    std::wstring GetTitle() const
    {
        return title;
    }

private:
    std::wstring title;
    std::wstring description;
    std::string file;
    unsigned int line;
};
