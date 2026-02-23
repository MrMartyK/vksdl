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

// Thin-film interference: wavelength-dependent phase shift.
vec3 thinFilmColor(float cosTheta, float thickness) {
    // Simplified Bragg reflection for 3 wavelength bands (RGB).
    float phase = thickness * cosTheta * 2.0;
    vec3 wavelengths = vec3(650.0, 510.0, 475.0); // nm (R, G, B approx)
    vec3 interference;
    interference.r = 0.5 + 0.5 * cos(phase * 6.283185 / wavelengths.r * 400.0);
    interference.g = 0.5 + 0.5 * cos(phase * 6.283185 / wavelengths.g * 400.0);
    interference.b = 0.5 + 0.5 * cos(phase * 6.283185 / wavelengths.b * 400.0);
    return interference;
}

void main() {
    vec3 N = normalize(cross(dFdx(vLocalPos), dFdy(vLocalPos)));
    vec3 V = normalize(-vLocalPos);

    float NdotV = max(dot(N, V), 0.0);

    // Noise-modulated film thickness for organic variation.
    float noiseVal = gradientNoise(vLocalPos * 8.0);
    float noiseVal2 = gradientNoise(vLocalPos * 16.0 + vec3(5.3));
    float thickness = 0.5 + 0.3 * noiseVal + 0.15 * noiseVal2;

    // Thin-film interference color.
    vec3 filmColor = thinFilmColor(NdotV, thickness);

    // Schlick Fresnel.
    float fresnel = pow(1.0 - NdotV, 5.0);
    float fresnelMix = 0.04 + 0.96 * fresnel;

    // Multi-band color shift for extra visual complexity.
    vec3 shiftedColor;
    shiftedColor.r = 0.5 + 0.5 * sin(NdotV * 12.0 + 0.0 + noiseVal * 3.0);
    shiftedColor.g = 0.5 + 0.5 * sin(NdotV * 12.0 + 2.094 + noiseVal * 3.0);
    shiftedColor.b = 0.5 + 0.5 * sin(NdotV * 12.0 + 4.189 + noiseVal * 3.0);

    // Blend base color, film interference, and angle-dependent shift.
    vec3 surfColor = mix(vColor.rgb, filmColor * shiftedColor, fresnelMix);

    // 2-light shading with specular highlights.
    vec3 L1 = normalize(vec3(1.0, 2.0, 3.0));
    vec3 L2 = normalize(vec3(-1.5, 1.0, -2.0));
    vec3 H1 = normalize(L1 + V);
    vec3 H2 = normalize(L2 + V);

    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0);
    float spec1 = pow(max(dot(N, H1), 0.0), 96.0);
    float spec2 = pow(max(dot(N, H2), 0.0), 64.0);

    vec3 ambient = surfColor * 0.08;
    vec3 diffuse = surfColor * (diff1 * 0.5 + diff2 * 0.3 + 0.1);
    vec3 specular = vec3(1.0, 0.95, 0.9) * spec1 * 0.6 + vec3(0.4, 0.5, 0.7) * spec2 * 0.3;

    // Rim glow.
    vec3 rimColor = filmColor * 0.5 * fresnel;

    outColor = vec4(ambient + diffuse + specular + rimColor, vColor.a);
}
