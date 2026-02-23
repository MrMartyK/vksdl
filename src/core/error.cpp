#include <vksdl/error.hpp>

#include <stdexcept>
#include <string>

namespace vksdl {

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
    throw std::runtime_error(e.format());
}

} // namespace vksdl
