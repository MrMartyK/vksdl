#version 460
#extension GL_EXT_ray_tracing : require

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

// Face normals indexed by (gl_InstanceCustomIndexEXT + gl_PrimitiveID).
// Convention: pyramid faces at [0..5] (customIndex=0),
//             ground faces at [6..7] (customIndex=6).
layout(binding = 2, set = 0) buffer Normals { vec4 normals[]; };

layout(push_constant) uniform PC {
    vec3 lightDir;
    float time;
    vec3 camPos;
    float camFov;
    vec3 camRight;
    float _pad1;
    vec3 camUp;
    float _pad2;
    vec3 camFwd;
    float _pad3;
};

struct RayPayload {
    vec3 color;
    int  depth;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT float shadowValue;

hitAttributeEXT vec2 attribs;

void main() {
    int currentDepth = payload.depth;

    // Face normal, transformed to world space for rotated instances.
    int normalIdx = gl_InstanceCustomIndexEXT + gl_PrimitiveID;
    vec3 objectNormal = normals[normalIdx].xyz;
    vec3 worldNormal = normalize(mat3(gl_ObjectToWorldEXT) * objectNormal);

    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    bool isPyramid = gl_InstanceCustomIndexEXT < 6;

    // --- Material ---
    vec3 material;
    if (isPyramid) {
        material = vec3(0.9, 0.7, 0.2);  // gold
    } else {
        // Checkerboard ground.
        float checker = mod(floor(hitPos.x) + floor(hitPos.z), 2.0);
        material = mix(vec3(0.35), vec3(0.65), checker);
    }

    // --- Hemisphere ambient ---
    vec3 skyAmbient    = vec3(0.06, 0.08, 0.15);
    vec3 groundAmbient = vec3(0.03, 0.02, 0.02);
    vec3 ambient = mix(groundAmbient, skyAmbient, worldNormal.y * 0.5 + 0.5);

    // --- Diffuse (Lambertian) ---
    float diffuse = max(dot(worldNormal, lightDir), 0.0);

    // --- Specular (Blinn-Phong) ---
    vec3 viewDir = normalize(-gl_WorldRayDirectionEXT);
    vec3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(worldNormal, halfVec), 0.0), 64.0);
    vec3 specColor = vec3(1.0, 0.85, 0.4);  // gold-tinted highlight

    // --- Shadow ray ---
    // Bias along normal to avoid self-intersection.
    shadowValue = 0.0;
    traceRayEXT(topLevelAS,
                gl_RayFlagsTerminateOnFirstHitEXT |
                gl_RayFlagsOpaqueEXT |
                gl_RayFlagsSkipClosestHitShaderEXT,
                0xFF,
                0, 0,
                1,  // missIndex=1 -> shadow_miss.rmiss
                hitPos + worldNormal * 0.01, 0.001, lightDir, 100.0,
                1);

    vec3 litColor = material * (ambient + diffuse * shadowValue)
                  + specColor * spec * shadowValue;

    // --- Reflection (pyramid/metal only, primary hits only) ---
    // Ground is diffuse -- no reflections. Pyramid is metallic gold.
    vec3 color;
    if (isPyramid && currentDepth < 1) {
        vec3 reflectDir = reflect(gl_WorldRayDirectionEXT, worldNormal);

        payload.color = vec3(0.0);
        payload.depth = currentDepth + 1;
        traceRayEXT(topLevelAS,
                     gl_RayFlagsOpaqueEXT,
                     0xFF,
                     0, 0,
                     0,  // missIndex=0 -> sky gradient
                     hitPos + worldNormal * 0.01, 0.001, reflectDir, 100.0,
                     0);
        vec3 reflected = payload.color;

        // Metallic blend: direct lighting tinted by material + strong reflection.
        color = litColor * 0.4 + reflected * 0.6;
    } else {
        color = litColor;
    }

    payload.color = color;
}
