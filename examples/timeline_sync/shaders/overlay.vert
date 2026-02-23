#version 450

vec2 positions[3] = vec2[](
    vec2( 0.0, -0.3),
    vec2(-0.3,  0.3),
    vec2( 0.3,  0.3)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 1.0, 1.0)
);

layout(location = 0) out vec3 fragColor;
layout(push_constant) uniform Push { float time; } pc;

void main() {
    vec2 pos   = positions[gl_VertexIndex];
    float angle = pc.time;
    float c = cos(angle);
    float s = sin(angle);
    pos = vec2(pos.x * c - pos.y * s, pos.x * s + pos.y * c);
    gl_Position = vec4(pos, 0.0, 1.0);
    fragColor = 0.5 + 0.5 * cos(pc.time + colors[gl_VertexIndex] * 6.28 + vec3(0, 2, 4));
}
