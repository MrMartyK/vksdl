#version 450

layout(location = 0) in vec3 vLocalPos;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 outColor;

// Hash for cell noise.
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

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

// 2D Voronoi distance (F1 and F2).
void voronoi(vec2 p, out float f1, out float f2, out vec2 cellId) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    f1 = 8.0;
    f2 = 8.0;
    cellId = vec2(0.0);

    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash2(n + g);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < f1) {
                f2 = f1;
                f1 = d;
                cellId = n + g;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    f1 = sqrt(f1);
    f2 = sqrt(f2);
}

void main() {
    vec3 N = normalize(cross(dFdx(vLocalPos), dFdy(vLocalPos)));
    vec3 V = normalize(-vLocalPos);

    // Voronoi pattern on two projected planes, blended by normal.
    vec2 uvXY = vLocalPos.xy * 4.0;
    vec2 uvXZ = vLocalPos.xz * 4.0;
    vec2 uvYZ = vLocalPos.yz * 4.0;

    float f1a, f2a, f1b, f2b, f1c, f2c;
    vec2 cellA, cellB, cellC;
    voronoi(uvXY, f1a, f2a, cellA);
    voronoi(uvXZ, f1b, f2b, cellB);
    voronoi(uvYZ, f1c, f2c, cellC);

    // Tri-planar blend weights.
    vec3 blend = abs(N);
    blend = blend / (blend.x + blend.y + blend.z);

    float f1 = f1a * blend.z + f1b * blend.y + f1c * blend.x;
    float f2 = f2a * blend.z + f2b * blend.y + f2c * blend.x;

    // Edge detection from F2-F1 difference.
    float edge = smoothstep(0.0, 0.15, f2 - f1);
    float cellShade = 0.4 + 0.6 * edge;

    // Noise distortion for organic feel.
    float noiseMod = gradientNoise(vLocalPos * 3.0) * 0.15;

    // Fresnel rim lighting.
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0);

    // 2-light shading.
    vec3 L1 = normalize(vec3(1.0, 2.0, 3.0));
    vec3 L2 = normalize(vec3(-1.5, 1.0, -2.0));
    float diff = max(dot(N, L1), 0.0) * 0.6 + max(dot(N, L2), 0.0) * 0.3 + 0.15;

    vec3 baseColor = vColor.rgb * (cellShade + noiseMod);
    vec3 result = baseColor * diff + vec3(0.6, 0.7, 1.0) * fresnel * 0.4;

    outColor = vec4(result, vColor.a);
}
