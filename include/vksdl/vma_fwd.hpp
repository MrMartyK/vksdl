#pragma once

// Forward-declare VMA types to avoid pulling vk_mem_alloc.h into user code.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;
