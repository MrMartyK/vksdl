#version 450

// Tonemap pass: reads HDR color, applies gamma correction, writes to swapchain.

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform sampler2D hdrColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 hdr = texture(hdrColor, inUV).rgb;

    // Reinhard tonemap + gamma.
    vec3 mapped = hdr / (hdr + vec3(1.0));
    vec3 gamma = pow(mapped, vec3(1.0 / 2.2));

    outColor = vec4(gamma, 1.0);
}
