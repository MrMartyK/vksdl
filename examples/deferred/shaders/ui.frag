#version 450

// UI pass: depth-aware overlay. Reads depth for testing, writes color.
// Draws a subtle vignette effect as a visual indicator.

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

void main() {
    // Vignette: darken edges for a simple UI overlay effect.
    float dist = length(inUV - vec2(0.5));
    float vignette = smoothstep(0.7, 0.3, dist);

    // Semi-transparent overlay (blend not enabled -- just write alpha).
    outColor = vec4(0.0, 0.0, 0.0, (1.0 - vignette) * 0.3);
}
