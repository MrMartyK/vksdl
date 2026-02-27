#include "mesh_loaders.hpp"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // unreferenced formal parameter
#pragma warning(disable : 4244) // conversion, possible loss of data
#pragma warning(disable : 4245) // signed/unsigned mismatch in initialization
#pragma warning(disable : 4505) // unreferenced local function
#pragma warning(disable : 4996) // 'fopen'/'strcpy'/'strncpy': unsafe CRT function
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vksdl::detail {

namespace {

// Generate flat normals for a triangle list. Vertices must already be unrolled
// (3 unique vertices per triangle, no sharing). Indices become 0..N-1.
void generateFlatNormals(std::vector<Vertex>& vertices) {
    for (std::size_t i = 0; i + 2 < vertices.size(); i += 3) {
        float ax = vertices[i + 1].position[0] - vertices[i].position[0];
        float ay = vertices[i + 1].position[1] - vertices[i].position[1];
        float az = vertices[i + 1].position[2] - vertices[i].position[2];
        float bx = vertices[i + 2].position[0] - vertices[i].position[0];
        float by = vertices[i + 2].position[1] - vertices[i].position[1];
        float bz = vertices[i + 2].position[2] - vertices[i].position[2];

        float nx = ay * bz - az * by;
        float ny = az * bx - ax * bz;
        float nz = ax * by - ay * bx;

        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-6f) {
            nx /= len;
            ny /= len;
            nz /= len;
        }

        for (std::size_t j = 0; j < 3; ++j) {
            vertices[i + j].normal[0] = nx;
            vertices[i + j].normal[1] = ny;
            vertices[i + j].normal[2] = nz;
        }
    }
}

} // anonymous namespace

Result<std::vector<MeshData>> loadGltf(const std::filesystem::path& path) {
    cgltf_options options{};
    cgltf_data* data = nullptr;

    std::string pathStr = path.string();

    cgltf_result res = cgltf_parse_file(&options, pathStr.c_str(), &data);
    if (res != cgltf_result_success) {
        return Error{"load model", 0, "failed to parse glTF file: " + pathStr};
    }

    res = cgltf_load_buffers(&options, data, pathStr.c_str());
    if (res != cgltf_result_success) {
        cgltf_free(data);
        return Error{"load model", 0, "failed to load glTF buffers: " + pathStr};
    }

    res = cgltf_validate(data);
    if (res != cgltf_result_success) {
        cgltf_free(data);
        return Error{"load model", 0, "glTF validation failed: " + pathStr};
    }

    std::filesystem::path parentDir = path.parent_path();

    std::vector<MeshData> meshes;

    for (cgltf_size mi = 0; mi < data->meshes_count; ++mi) {
        const cgltf_mesh& gltfMesh = data->meshes[mi];

        for (cgltf_size pi = 0; pi < gltfMesh.primitives_count; ++pi) {
            const cgltf_primitive& prim = gltfMesh.primitives[pi];

            if (prim.type != cgltf_primitive_type_triangles) {
                continue; // skip non-triangle primitives
            }

            // Find accessors by iterating attributes
            const cgltf_accessor* posAccessor = nullptr;
            const cgltf_accessor* normAccessor = nullptr;
            const cgltf_accessor* uvAccessor = nullptr;

            for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
                const cgltf_attribute& attr = prim.attributes[ai];
                if (attr.type == cgltf_attribute_type_position) {
                    posAccessor = attr.data;
                } else if (attr.type == cgltf_attribute_type_normal) {
                    normAccessor = attr.data;
                } else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) {
                    uvAccessor = attr.data;
                }
            }

            if (!posAccessor) {
                continue; // no positions, skip
            }

            bool hasNormals = normAccessor != nullptr;
            bool hasUVs = uvAccessor != nullptr;

            // Read indices
            std::vector<std::uint32_t> indices;
            if (prim.indices) {
                indices.resize(prim.indices->count);
                cgltf_accessor_unpack_indices(prim.indices, indices.data(), sizeof(std::uint32_t),
                                              prim.indices->count);
            } else {
                // Non-indexed: generate sequential indices
                indices.resize(posAccessor->count);
                for (cgltf_size i = 0; i < posAccessor->count; ++i) {
                    indices[i] = static_cast<std::uint32_t>(i);
                }
            }

            // Read vertex data
            std::size_t vertexCount = posAccessor->count;
            std::vector<float> positions(vertexCount * 3);
            cgltf_accessor_unpack_floats(posAccessor, positions.data(), vertexCount * 3);

            std::vector<float> normals;
            if (hasNormals) {
                normals.resize(vertexCount * 3);
                cgltf_accessor_unpack_floats(normAccessor, normals.data(), vertexCount * 3);
            }

            std::vector<float> uvs;
            if (hasUVs) {
                uvs.resize(vertexCount * 2);
                cgltf_accessor_unpack_floats(uvAccessor, uvs.data(), vertexCount * 2);
            }

            // Build interleaved vertices
            if (!hasNormals) {
                // Must de-index to generate flat normals: unroll vertices
                // per triangle so each triangle has 3 unique vertices.
                std::vector<Vertex> unrolled;
                unrolled.reserve(indices.size());

                for (std::uint32_t idx : indices) {
                    if (static_cast<std::size_t>(idx) >= vertexCount)
                        continue;
                    Vertex v{};
                    v.position[0] = positions[idx * 3 + 0];
                    v.position[1] = positions[idx * 3 + 1];
                    v.position[2] = positions[idx * 3 + 2];
                    if (hasUVs) {
                        v.texCoord[0] = uvs[idx * 2 + 0];
                        v.texCoord[1] = uvs[idx * 2 + 1];
                    }
                    unrolled.push_back(v);
                }

                generateFlatNormals(unrolled);

                // Rebuild sequential indices
                std::vector<std::uint32_t> newIndices(unrolled.size());
                for (std::size_t i = 0; i < unrolled.size(); ++i) {
                    newIndices[i] = static_cast<std::uint32_t>(i);
                }

                MeshData meshData;
                meshData.vertices = std::move(unrolled);
                meshData.indices = std::move(newIndices);

                if (gltfMesh.name) {
                    meshData.name = gltfMesh.name;
                }

                // Material
                if (prim.material) {
                    if (prim.material->name) {
                        meshData.material.name = prim.material->name;
                    }
                    if (prim.material->has_pbr_metallic_roughness) {
                        const auto& pbr = prim.material->pbr_metallic_roughness;
                        std::memcpy(meshData.material.baseColor, pbr.base_color_factor,
                                    sizeof(float) * 4);
                        meshData.material.metallic = pbr.metallic_factor;
                        meshData.material.roughness = pbr.roughness_factor;

                        if (pbr.base_color_texture.texture &&
                            pbr.base_color_texture.texture->image &&
                            pbr.base_color_texture.texture->image->uri) {
                            meshData.material.diffuseTexture =
                                parentDir / pbr.base_color_texture.texture->image->uri;
                        }
                    }
                }

                meshes.push_back(std::move(meshData));

            } else {
                // Have normals -- build vertices directly
                std::vector<Vertex> verts(vertexCount);
                for (std::size_t i = 0; i < vertexCount; ++i) {
                    verts[i].position[0] = positions[i * 3 + 0];
                    verts[i].position[1] = positions[i * 3 + 1];
                    verts[i].position[2] = positions[i * 3 + 2];
                    verts[i].normal[0] = normals[i * 3 + 0];
                    verts[i].normal[1] = normals[i * 3 + 1];
                    verts[i].normal[2] = normals[i * 3 + 2];
                    if (hasUVs) {
                        verts[i].texCoord[0] = uvs[i * 2 + 0];
                        verts[i].texCoord[1] = uvs[i * 2 + 1];
                    }
                }

                MeshData meshData;
                meshData.vertices = std::move(verts);
                meshData.indices = std::move(indices);

                if (gltfMesh.name) {
                    meshData.name = gltfMesh.name;
                }

                // Material
                if (prim.material) {
                    if (prim.material->name) {
                        meshData.material.name = prim.material->name;
                    }
                    if (prim.material->has_pbr_metallic_roughness) {
                        const auto& pbr = prim.material->pbr_metallic_roughness;
                        std::memcpy(meshData.material.baseColor, pbr.base_color_factor,
                                    sizeof(float) * 4);
                        meshData.material.metallic = pbr.metallic_factor;
                        meshData.material.roughness = pbr.roughness_factor;

                        if (pbr.base_color_texture.texture &&
                            pbr.base_color_texture.texture->image &&
                            pbr.base_color_texture.texture->image->uri) {
                            meshData.material.diffuseTexture =
                                parentDir / pbr.base_color_texture.texture->image->uri;
                        }
                    }
                }

                meshes.push_back(std::move(meshData));
            }
        }
    }

    cgltf_free(data);

    if (meshes.empty()) {
        return Error{"load model", 0, "no triangle meshes found in: " + pathStr};
    }

    return meshes;
}

} // namespace vksdl::detail
