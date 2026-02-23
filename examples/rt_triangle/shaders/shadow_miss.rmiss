#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT float shadowValue;

void main() {
    // Ray missed all geometry -- point is lit (not in shadow).
    shadowValue = 1.0;
}
