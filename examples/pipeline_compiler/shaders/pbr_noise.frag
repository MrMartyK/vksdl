#version 450

layout(location = 0) in vec3 vLocalPos;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 outColor;

// Hash-based gradient noise (no textures needed).
vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123) * 2.0 - 1.0;
}

float gradientNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    return mix(mix(mix(dot(hash3(i + vec3(0, 0, 0)), f - vec3(0, 0, 0)),
                       dot(hash3(i + vec3(1, 0, 0)), f - vec3(1, 0, 0)), u.x),
                   mix(dot(hash3(i + vec3(0, 1, 0)), f - vec3(0, 1, 0)),
                       dot(hash3(i + vec3(1, 1, 0)), f - vec3(1, 1, 0)), u.x), u.y),
               mix(mix(dot(hash3(i + vec3(0, 0, 1)), f - vec3(0, 0, 1)),
                       dot(hash3(i + vec3(1, 0, 1)), f - vec3(1, 0, 1)), u.x),
                   mix(dot(hash3(i + vec3(0, 1, 1)), f - vec3(0, 1, 1)),
                       dot(hash3(i + vec3(1, 1, 1)), f - vec3(1, 1, 1)), u.x), u.y), u.z);
}

float fbm(vec3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * gradientNoise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

void main() {
    // Flat normal from screen-space derivatives.
    vec3 N = normalize(cross(dFdx(vLocalPos), dFdy(vLocalPos)));

    // Noise-based surface detail.
    float noise = fbm(vLocalPos * 6.0, 6);
    float roughness = 0.3 + 0.4 * (noise * 0.5 + 0.5);

    // 4 directional lights with diffuse + Blinn-Phong specular.
    vec3 lights[4] = vec3[4](
        normalize(vec3( 1.0,  2.0,  3.0)),
        normalize(vec3(-2.0,  1.0, -1.0)),
        normalize(vec3( 0.5, -1.0,  2.0)),
        normalize(vec3(-1.0,  3.0, -2.0))
    );
    vec3 lightColors[4] = vec3[4](
        vec3(1.0, 0.95, 0.9),
        vec3(0.4, 0.5, 0.7),
        vec3(0.3, 0.6, 0.3),
        vec3(0.7, 0.4, 0.3)
    );

    vec3 V = normalize(-vLocalPos);
    vec3 baseColor = vColor.rgb * (0.7 + 0.3 * noise);
    vec3 result = baseColor * 0.05; // ambient

    for (int i = 0; i < 4; ++i) {
        vec3 L = lights[i];
        vec3 H = normalize(L + V);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float spec = pow(NdotH, mix(16.0, 128.0, 1.0 - roughness));

        result += lightColors[i] * 0.3 * (baseColor * NdotL + vec3(spec) * 0.5);
    }

    outColor = vec4(result, vColor.a);
}
