#pragma once

#include <cstdint>

namespace vksdl {

// Policy for pipeline creation strategy selection.
// Auto selects the best available path based on device capabilities.
enum class PipelinePolicy : std::uint8_t {
    Auto,            // GPL+fast-link if available, else monolithic
    ForceMonolithic, // Always monolithic (current behavior)
    PreferGPL,       // Force GPL path even without fast-linking
    // Reserved for VK_EXT_shader_object support (not yet implemented).
    // Selecting this policy returns an error until the extension is supported.
    ForceShaderObject,
};

// Resolved pipeline creation model (result of applying policy to capabilities).
enum class PipelineModel : std::uint8_t {
    Monolithic,
    GPL,
};

// Information about the resolved pipeline model. Returned by PipelineCompiler
// for diagnostic/logging purposes.
struct PipelineModelInfo {
    PipelineModel model = PipelineModel::Monolithic;
    bool hasPCCC = false; // cache probe available
    bool hasGPL = false;
    bool fastLink = false; // driver supports fast linking
};

} // namespace vksdl
