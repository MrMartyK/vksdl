#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    int  depth;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

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

void main() {
    float t = 0.5 * (gl_WorldRayDirectionEXT.y + 1.0);
    vec3 sky = mix(vec3(0.08, 0.08, 0.12), vec3(0.1, 0.15, 0.35), t);

    // Sun orb: bright core with soft glow falloff.
    float sunDot = dot(normalize(gl_WorldRayDirectionEXT), lightDir);
    float core = smoothstep(0.995, 0.999, sunDot);  // hard bright center
    float glow = smoothstep(0.96, 0.995, sunDot);   // soft halo around it

    vec3 sunColor = vec3(1.0, 0.95, 0.8);
    sky += sunColor * core * 2.0;   // bright white-gold core
    sky += sunColor * glow * 0.3;   // warm glow ring

    payload.color = sky;
}
