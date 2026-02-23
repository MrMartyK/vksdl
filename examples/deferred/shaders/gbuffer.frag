#version 450

// G-buffer pass: writes albedo and normals to two color attachments.

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormals;

void main() {
    // Constant albedo: mid-blue surface.
    outAlbedo = vec4(0.5, 0.5, 1.0, 1.0);

    // Constant normals: pointing up (0,0,1) in view space.
    outNormals = vec4(0.0, 0.0, 1.0, 0.0);
}
