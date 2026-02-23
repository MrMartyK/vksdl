#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp;
    vec4 color; // w unused; vec4 for std140 alignment
} ubo;

layout(location = 0) in vec3 inPos;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.mvp * vec4(inPos, 1.0);

    // Approximate face normal from vertex position for simple diffuse shading.
    vec3 n = normalize(inPos);
    float light = 0.25 + 0.75 * max(dot(n, normalize(vec3(1.0, 2.0, 3.0))), 0.0);
    fragColor = ubo.color.rgb * light;
}
