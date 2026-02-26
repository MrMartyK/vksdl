#include <vksdl/error.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace vksdl {

#ifndef VKSDL_ENABLE_EXCEPTIONS
#define VKSDL_ENABLE_EXCEPTIONS 1
#endif

std::string Error::format() const {
    std::string out = "vksdl: " + operation + " failed";

    if (vkResult != 0) {
        out += " (VkResult " + std::to_string(vkResult) + ")";
    }

    if (!message.empty()) {
        out += ": " + message;
    }

    return out;
}

void throwError(const Error& e) {
#if VKSDL_ENABLE_EXCEPTIONS
    throw std::runtime_error(e.format());
#else
    std::fprintf(stderr, "%s\n", e.format().c_str());
    std::abort();
#endif
}

} // namespace vksdl
