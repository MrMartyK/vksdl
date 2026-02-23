#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Material {
    vec4 baseColor;
} material;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(normalize(fragNormal), lightDir), 0.0);
    float ambient = 0.15;
    vec3 color = material.baseColor.rgb * (ambient + diffuse);
    outColor = vec4(color, material.baseColor.a);
}
