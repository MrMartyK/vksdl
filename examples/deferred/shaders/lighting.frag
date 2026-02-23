#version 450

// Lighting pass: reads G-buffer albedo, normals, depth, and shadow map.
// Writes tinted composite to HDR output.

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform sampler2D shadowDepth;
layout(set = 0, binding = 1) uniform sampler2D gbufAlbedo;
layout(set = 0, binding = 2) uniform sampler2D gbufNormals;
layout(set = 0, binding = 3) uniform sampler2D gbufDepth;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 albedo  = texture(gbufAlbedo,  inUV);
    vec4 normals = texture(gbufNormals, inUV);
    float shadow = texture(shadowDepth, inUV).r;
    float depth  = texture(gbufDepth,   inUV).r;

    // Simple directional light from above-right.
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float NdotL = max(dot(normals.xyz, lightDir), 0.0);

    // Tint: warm light color * shadow term.
    vec3 lit = albedo.rgb * NdotL * vec3(1.0, 0.9, 0.8);

    // Ambient term.
    vec3 ambient = albedo.rgb * 0.15;

    outColor = vec4(lit + ambient, 1.0);
}
