#pragma once

// Render graph umbrella header.
// Separate from <vksdl/vksdl.hpp> -- include this only when using the
// render graph. Core users never see graph types.

#include <vksdl/graph/barrier_compiler.hpp>
#include <vksdl/graph/pass.hpp>
#include <vksdl/graph/pass_context.hpp>
#include <vksdl/graph/render_graph.hpp>
#include <vksdl/graph/resource.hpp>
#include <vksdl/graph/resource_state.hpp>
