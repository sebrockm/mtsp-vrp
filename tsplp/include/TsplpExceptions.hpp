#pragma once

#include <exception>

namespace tsplp
{
    class TsplpException : public std::exception
    {
    };

    class CyclicDependenciesException : public TsplpException
    {
    };

    class IncompatibleDependenciesException : public TsplpException
    {
    };
}
