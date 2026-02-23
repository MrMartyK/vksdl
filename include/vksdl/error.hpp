#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace vksdl {

// Thin error type that carries what we tried, what Vulkan said, and a human message.
// VkResult is stored as int32_t to avoid pulling <vulkan/vulkan.h> into every header.
struct Error {
    std::string   operation;  // e.g. "create device"
    std::int32_t  vkResult;   // 0 (VK_SUCCESS) when not a Vulkan error
    std::string   message;    // human-readable explanation + suggestion

    // Format as a single readable string.
    [[nodiscard]] std::string format() const;
};

// Throw an Error as std::runtime_error. Defined in error.cpp so <stdexcept>
// is not pulled into every translation unit via result.hpp.
[[noreturn]] void throwError(const Error& e);

} // namespace vksdl
