#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3  attenuation;
    vec3  scatterOrigin;
    vec3  scatterDir;
    uint  seed;
    bool  missed;
    vec3  color;
    vec3  directLight;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

// Must match closesthit.rchit.
const float PI      = 3.14159265359;
const vec3  sunDir   = normalize(vec3(0.6, 0.8, -0.3));
const vec3  sunColor = vec3(6.0, 5.6, 4.8);

// ---------------------------------------------------------------------------
// Analytic atmospheric sky (Rayleigh + Mie approximation)
//
// Inspired by Preetham/Hosek-Wilkie. Derives sky color entirely from sunDir
// -- no textures, no lookup tables. Outputs HDR; tone mapping in raygen.
// ---------------------------------------------------------------------------

vec3 atmosphericSky(vec3 dir) {
    float cosGamma = dot(dir, sunDir);
    float gamma    = acos(clamp(cosGamma, -1.0, 1.0));

    // Zenith luminance varies with sun elevation.
    float sunY   = sunDir.y;
    float zenithBrightness = max(0.1, 0.8 + 0.6 * sunY);

    // Rayleigh scattering: blue overhead, warm at horizon.
    // Phase function ~ (1 + cos^2(gamma)).
    float rayleighPhase = 0.75 * (1.0 + cosGamma * cosGamma);

    // Zenith color: deep blue when sun is high, desaturated near sunset.
    vec3 zenithColor = mix(vec3(0.15, 0.3, 0.8), vec3(0.25, 0.45, 0.9), clamp(sunY, 0.0, 1.0));
    zenithColor *= zenithBrightness;

    // Altitude-dependent gradient: bluer overhead, warmer at horizon.
    float altitude = max(dir.y, 0.0);
    float horizonFade = 1.0 - pow(altitude, 0.4);

    // Horizon color: warm glow from Rayleigh extinction (red survives long paths).
    vec3 horizonColor = vec3(0.9, 0.75, 0.55) * zenithBrightness * 0.7;

    vec3 sky = mix(zenithColor, horizonColor, horizonFade);
    sky *= rayleighPhase * 0.5 + 0.5; // Modulate by Rayleigh phase.

    // Mie scattering: forward-peaked aureole around the sun.
    // Henyey-Greenstein approximation with g ~ 0.76.
    float g = 0.76;
    float g2 = g * g;
    float miePhase = (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosGamma, 1.5));
    vec3 mieColor = sunColor * 0.15 * miePhase;
    sky += mieColor;

    // Sun disk: bright hotspot visible in reflections.
    sky += sunColor * smoothstep(0.9995, 0.99995, cosGamma);

    // Below-horizon darkening.
    if (dir.y < 0.0) {
        sky *= exp(dir.y * 8.0); // Exponential falloff below horizon.
    }

    return sky;
}

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    payload.color       = atmosphericSky(dir);
    payload.directLight = vec3(0.0);
    payload.missed      = true;
}
