#include "mesh_loaders.hpp"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // unreferenced formal parameter
#pragma warning(disable : 4244) // conversion, possible loss of data
#pragma warning(disable : 4245) // signed/unsigned mismatch in initialization
#pragma warning(disable : 4996) // unsafe CRT function
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cmath>
#include <unordered_map>

namespace vksdl::detail {

namespace {

// Generate flat normals for a triangle list of unrolled vertices.
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

        for (int j = 0; j < 3; ++j) {
            vertices[i + j].normal[0] = nx;
            vertices[i + j].normal[1] = ny;
            vertices[i + j].normal[2] = nz;
        }
    }
}

} // anonymous namespace

Result<std::vector<MeshData>> loadObj(const std::filesystem::path& path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    std::string pathStr = path.string();
    std::string mtlDir = path.parent_path().string();

    bool ok =
        tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, pathStr.c_str(), mtlDir.c_str(),
                         /*triangulate=*/true);
    if (!ok) {
        std::string msg = "failed to load OBJ: " + pathStr;
        if (!err.empty()) {
            msg += " -- " + err;
        }
        return Error{"load model", 0, msg};
    }

    bool hasNormals = !attrib.normals.empty();
    bool hasUVs = !attrib.texcoords.empty();

    std::filesystem::path parentDir = path.parent_path();

    std::vector<MeshData> meshes;

    for (const auto& shape : shapes) {
        // OBJ shapes can reference different materials per face.
        // Group faces by material ID to create separate MeshData per material.
        std::unordered_map<int, std::vector<std::size_t>> facesByMaterial;
        std::size_t faceOffset = 0;

        for (std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int matId = shape.mesh.material_ids[f];
            facesByMaterial[matId].push_back(faceOffset);
            faceOffset += shape.mesh.num_face_vertices[f];
        }

        for (auto& [matId, faceStarts] : facesByMaterial) {
            std::vector<Vertex> vertices;

            // Each face start points to the first index of a triangulated face (3 verts).
            // We unroll all vertices per triangle for flat normal generation.
            for (std::size_t start : faceStarts) {
                for (int v = 0; v < 3; ++v) {
                    const tinyobj::index_t& idx = shape.mesh.indices[start + v];

                    Vertex vert{};
                    if (idx.vertex_index < 0 ||
                        static_cast<std::size_t>(3 * idx.vertex_index + 2) >=
                            attrib.vertices.size())
                        continue;
                    vert.position[0] = attrib.vertices[3 * idx.vertex_index + 0];
                    vert.position[1] = attrib.vertices[3 * idx.vertex_index + 1];
                    vert.position[2] = attrib.vertices[3 * idx.vertex_index + 2];

                    if (hasNormals && idx.normal_index >= 0 &&
                        static_cast<std::size_t>(3 * idx.normal_index + 2) <
                            attrib.normals.size()) {
                        vert.normal[0] = attrib.normals[3 * idx.normal_index + 0];
                        vert.normal[1] = attrib.normals[3 * idx.normal_index + 1];
                        vert.normal[2] = attrib.normals[3 * idx.normal_index + 2];
                    }

                    if (hasUVs && idx.texcoord_index >= 0 &&
                        static_cast<std::size_t>(2 * idx.texcoord_index + 1) <
                            attrib.texcoords.size()) {
                        vert.texCoord[0] = attrib.texcoords[2 * idx.texcoord_index + 0];
                        vert.texCoord[1] = attrib.texcoords[2 * idx.texcoord_index + 1];
                    }

                    vertices.push_back(vert);
                }
            }

            if (vertices.empty()) {
                continue;
            }

            // Generate flat normals if the OBJ didn't have normals
            if (!hasNormals) {
                generateFlatNormals(vertices);
            }

            // Build sequential indices (already unrolled)
            std::vector<std::uint32_t> indices(vertices.size());
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                indices[i] = static_cast<std::uint32_t>(i);
            }

            MeshData meshData;
            meshData.vertices = std::move(vertices);
            meshData.indices = std::move(indices);
            meshData.name = shape.name;

            // Material
            if (matId >= 0 && matId < static_cast<int>(materials.size())) {
                const auto& mat = materials[matId];
                meshData.material.name = mat.name;
                meshData.material.baseColor[0] = mat.diffuse[0];
                meshData.material.baseColor[1] = mat.diffuse[1];
                meshData.material.baseColor[2] = mat.diffuse[2];
                meshData.material.baseColor[3] = 1.0f - (1.0f - mat.dissolve);
                meshData.material.metallic = mat.specular[0]; // rough approximation
                meshData.material.roughness = 1.0f - (mat.shininess / 1000.0f);
                if (meshData.material.roughness < 0.0f) {
                    meshData.material.roughness = 0.0f;
                }

                if (!mat.diffuse_texname.empty()) {
                    meshData.material.diffuseTexture = parentDir / mat.diffuse_texname;
                }
            }

            meshes.push_back(std::move(meshData));
        }
    }

    if (meshes.empty()) {
        return Error{"load model", 0, "no meshes found in OBJ: " + pathStr};
    }

    return meshes;
}

} // namespace vksdl::detail
