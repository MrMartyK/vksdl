#version 450

layout(location = 0) in vec3 vLocalPos;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 outColor;

// Hash-based gradient noise.
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

float turbulence(vec3 p, int octaves) {
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float totalAmplitude = 0.0;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * abs(gradientNoise(p * frequency));
        totalAmplitude += amplitude;
        amplitude *= 0.5;
        frequency *= 2.1;
    }
    return value / totalAmplitude;
}

void main() {
    vec3 N = normalize(cross(dFdx(vLocalPos), dFdy(vLocalPos)));
    vec3 V = normalize(-vLocalPos);

    // Marble vein pattern: turbulence modulates a sin wave.
    vec3 p = vLocalPos * 5.0;
    float turb = turbulence(p, 7);
    float vein = sin(p.x * 2.0 + p.y * 3.0 + p.z * 1.5 + turb * 8.0);
    vein = 0.5 + 0.5 * vein; // remap to [0, 1]

    // Secondary vein layer at different scale.
    float turb2 = turbulence(p * 1.7 + vec3(17.3), 5);
    float vein2 = sin(p.y * 4.0 - p.z * 2.0 + turb2 * 6.0);
    vein2 = 0.5 + 0.5 * vein2;

    // Blend veins with base color.
    vec3 veinColor = vColor.rgb * 0.15;
    vec3 baseColor = vColor.rgb;
    vec3 surfColor = mix(veinColor, baseColor, vein * 0.7 + vein2 * 0.3);

    // Subsurface scattering approximation: warm color bleed at grazing angles.
    float subsurface = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 sssColor = vColor.rgb * vec3(1.2, 0.8, 0.6) * subsurface * 0.3;

    // 3-light directional shading.
    vec3 lights[3] = vec3[3](
        normalize(vec3( 1.0, 2.0,  3.0)),
        normalize(vec3(-2.0, 1.0, -1.0)),
        normalize(vec3( 0.0, 3.0,  0.0))
    );
    vec3 lightCols[3] = vec3[3](
        vec3(1.0, 0.95, 0.9),
        vec3(0.3, 0.4, 0.6),
        vec3(0.5, 0.5, 0.5)
    );

    vec3 result = surfColor * 0.08; // ambient
    for (int i = 0; i < 3; ++i) {
        vec3 L = lights[i];
        vec3 H = normalize(L + V);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float spec = pow(NdotH, 64.0) * (1.0 - turb * 0.5);
        result += lightCols[i] * 0.35 * (surfColor * NdotL + vec3(spec) * 0.3);
    }

    result += sssColor;

    outColor = vec4(result, vColor.a);
}
