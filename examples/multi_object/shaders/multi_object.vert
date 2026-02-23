#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
} scene;

layout(set = 0, binding = 1) uniform ObjectUBO {
    mat4 model;
} object;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = scene.proj * scene.view * object.model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
}
