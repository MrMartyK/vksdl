#version 460
#extension GL_EXT_ray_tracing : require

struct Material {
    vec3  albedo;
    uint  type;          // 0=diffuse, 1=metal, 2=dielectric
    float roughness;     // metal fuzz
    float ior;           // dielectric IOR (1.5 for glass)
    float specRoughness; // GGX roughness for diffuse specular lobe
    float absorption;    // Beer-Lambert scale for dielectrics (0 = clear)
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 3, set = 0) buffer Materials { Material materials[]; };

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

hitAttributeEXT vec2 attribs;

// --- Constants ---

const float PI = 3.14159265359;
const vec3  sunDir   = normalize(vec3(0.6, 0.8, -0.3));
const vec3  sunColor = vec3(6.0, 5.6, 4.8);  // warm directional

// --- RNG (must match raygen) ---

uint pcg(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(inout uint state) {
    return float(pcg(state)) / 4294967295.0;
}

vec3 randomInUnitSphere(inout uint state) {
    vec3 p;
    for (int i = 0; i < 16; i++) {
        p = vec3(randomFloat(state), randomFloat(state), randomFloat(state)) * 2.0 - 1.0;
        if (dot(p, p) < 1.0) return p;
    }
    return normalize(p);
}

vec3 randomUnitVector(inout uint state) {
    return normalize(randomInUnitSphere(state));
}

// Schlick approximation for Fresnel reflectance.
float schlick(float cosine, float ri) {
    float r0 = (1.0 - ri) / (1.0 + ri);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

// Schlick Fresnel with explicit F0.
float schlickF0(float cosTheta, float f0) {
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// GGX/Trowbridge-Reitz normal distribution function.
float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith G2 height-correlated visibility (combined G / (4 * NdotV * NdotL)).
float V_SmithGGX(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(ggxV + ggxL, 1e-7);
}

// Trace shadow ray toward sun. Returns true if sun is visible.
bool traceShadowRay(vec3 origin, vec3 normal) {
    // Save payload state (shadow trace overwrites it).
    vec3 savedAtt  = payload.attenuation;
    vec3 savedOrig = payload.scatterOrigin;
    vec3 savedDir  = payload.scatterDir;
    uint savedSeed = payload.seed;
    vec3 savedCol  = payload.color;
    vec3 savedDL   = payload.directLight;

    payload.missed = false;
    traceRayEXT(tlas,
                gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
                0xFF, 0, 0, 0,
                origin + normal * 0.002, 0.001, sunDir, 10000.0, 0);

    bool visible = payload.missed;

    // Restore payload.
    payload.attenuation   = savedAtt;
    payload.scatterOrigin = savedOrig;
    payload.scatterDir    = savedDir;
    payload.seed          = savedSeed;
    payload.missed        = false;
    payload.color         = savedCol;
    payload.directLight   = savedDL;

    return visible;
}

void main() {
    Material mat = materials[gl_InstanceCustomIndexEXT];

    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Compute surface normal based on geometry type.
    vec3 worldNormal;
    if (gl_InstanceCustomIndexEXT == 0u) {
        // Ground plane -- always faces up.
        worldNormal = vec3(0.0, 1.0, 0.0);

        // Checker pattern for spatial context.
        float check = mod(floor(hitPos.x) + floor(hitPos.z), 2.0);
        mat.albedo = mix(vec3(0.3), vec3(0.75), check);
    } else {
        // Icosphere -- object-space hit position IS the normal for a unit sphere.
        vec3 objHit = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_HitTEXT;
        worldNormal = normalize(mat3(gl_ObjectToWorldEXT) * normalize(objHit));
    }

    // Ensure normal faces the incoming ray.
    bool frontFace = dot(gl_WorldRayDirectionEXT, worldNormal) < 0.0;
    vec3 outwardNormal = frontFace ? worldNormal : -worldNormal;

    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);

    payload.directLight = vec3(0.0);

    if (mat.type == 0u) {
        // --- Diffuse with GGX specular lobe ---
        // Cosine-weighted hemisphere sampling for the diffuse component.
        // GGX specular evaluated analytically via NEE (sun sampling).
        // NOTE: indirect specular from this lobe is importance-sampled by
        // the cosine PDF, which works well for specRoughness >= 0.3. Lower
        // values would need GGX importance sampling for acceptable noise.
        vec3 scatterDir = outwardNormal + randomUnitVector(payload.seed);
        if (dot(scatterDir, scatterDir) < 1e-8)
            scatterDir = outwardNormal;

        payload.attenuation   = mat.albedo;
        payload.scatterOrigin = hitPos + outwardNormal * 0.001;
        payload.scatterDir    = normalize(scatterDir);
        payload.missed        = false;

        // Next-event estimation with energy-conserving BRDF.
        float NdotL = dot(outwardNormal, sunDir);
        if (NdotL > 0.0 && traceShadowRay(hitPos, outwardNormal)) {
            vec3 V = -rayDir;
            vec3 H = normalize(V + sunDir);
            float NdotV = max(dot(outwardNormal, V), 1e-4);
            float NdotH = max(dot(outwardNormal, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);

            // Dielectric Fresnel: F0 ~ 0.04 (IOR ~1.5).
            float F = schlickF0(VdotH, 0.04);

            // GGX specular: alpha = roughness^2.
            float alpha = mat.specRoughness * mat.specRoughness;
            float D = D_GGX(NdotH, alpha);
            float Vis = V_SmithGGX(NdotV, NdotL, alpha);

            // Energy-conserving: diffuse dims as specular increases.
            vec3 diffuse = (1.0 - F) * mat.albedo / PI;
            vec3 spec    = vec3(F * D * Vis);

            payload.directLight = (diffuse + spec) * sunColor * NdotL;
        }

    } else if (mat.type == 1u) {
        // --- Metal ---
        vec3 reflected = reflect(rayDir, outwardNormal);
        reflected = normalize(reflected + mat.roughness * randomInUnitSphere(payload.seed));

        payload.attenuation   = mat.albedo;
        payload.scatterOrigin = hitPos + outwardNormal * 0.001;
        payload.scatterDir    = reflected;
        payload.missed        = false;

        // Absorb if scattered below surface.
        if (dot(reflected, outwardNormal) <= 0.0) {
            payload.attenuation = vec3(0.0);
        } else {
            // Sun specular highlight for metals.
            float NdotL = dot(outwardNormal, sunDir);
            if (NdotL > 0.0 && traceShadowRay(hitPos, outwardNormal)) {
                // Phong-like specular from sun.
                vec3 sunReflect = reflect(-sunDir, outwardNormal);
                float spec = pow(max(dot(sunReflect, -rayDir), 0.0), mix(32.0, 256.0, 1.0 - mat.roughness));
                payload.directLight = mat.albedo * sunColor * spec;
            }
        }

    } else {
        // --- Dielectric (glass) with Beer-Lambert absorption ---
        float eta = frontFace ? (1.0 / mat.ior) : mat.ior;
        float cosTheta = min(dot(-rayDir, outwardNormal), 1.0);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = eta * sinTheta > 1.0;
        float reflectProb = schlick(cosTheta, eta);

        vec3 scattered;
        if (cannotRefract || randomFloat(payload.seed) < reflectProb) {
            scattered = reflect(rayDir, outwardNormal);
        } else {
            scattered = refract(rayDir, outwardNormal, eta);
        }

        // Beer-Lambert: absorb light as it travels through the medium.
        // Applied on exit (back-face hit = ray traveled inside the material).
        vec3 att = vec3(1.0);
        if (!frontFace && mat.absorption > 0.0) {
            // sigma_a derived from albedo: albedo is the "tint at reference distance".
            // -log(albedo) gives per-unit absorption; absorption field scales intensity.
            vec3 sigma_a = max(-log(max(mat.albedo, vec3(1e-6))) * mat.absorption, vec3(0.0));
            att = exp(-sigma_a * gl_HitTEXT);
        }

        payload.attenuation   = att;
        payload.scatterOrigin = hitPos + scattered * 0.001;
        payload.scatterDir    = normalize(scattered);
        payload.missed        = false;
    }
}
