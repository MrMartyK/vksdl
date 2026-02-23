#version 450

// Shadow pass: depth-only, write no color.
// Fragment shader exists only to set depth value.

void main() {
    // Fullscreen triangle at depth 0.5 (arbitrary, just populates the map).
    gl_FragDepth = 0.5;
}
