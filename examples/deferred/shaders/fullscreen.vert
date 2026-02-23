#version 450

// Fullscreen triangle without vertex buffer.
// Generates vertices for a triangle that covers the entire screen.
// Draw with vkCmdDraw(cmd, 3, 1, 0, 0).

layout(location = 0) out vec2 outUV;

void main() {
    // Vertices: (-1,-1), (3,-1), (-1,3) -- covers full NDC quad.
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
