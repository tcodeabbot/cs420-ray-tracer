// main_gpu.cu - Week 2: CUDA GPU Ray Tracer
// CS420 Ray Tracer Project
//
// GPU IMPLEMENTATION FEATURES:
// ============================
// Part A - Basic GPU Implementation:
//   ✓ Ray-sphere intersection on GPU (iterative, not recursive)
//   ✓ Thread mapping: one thread per pixel
//   ✓ Proper memory management (cudaMalloc/cudaMemcpy/cudaFree)
//   ✓ Scene loading from file
//
// Part B - GPU Optimizations:
//   ✓ Shared memory for sphere data (cooperative loading)
//   ✓ Constant memory for lights and ambient (broadcast to all threads)
//   ✓ Anti-aliasing with multiple samples per pixel (cuRAND)
//   ✓ Block size optimized for warp efficiency (16x16 = 256 threads)
//   ✓ Coalesced memory access patterns
//
// Performance Features:
//   - Handles 100+ spheres efficiently
//   - Targets 10x+ speedup over serial implementation
//   - No register spilling (efficient kernel code)
//   - Proper bounds checking to avoid memory errors

#include "cuda_fix.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

// Include math constants for cross-platform compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// =========================================================
// GPU Vector and Ray Classes (simplified for CUDA)
// =========================================================

struct float3_ops {
    __host__ __device__ static float3 make(float x, float y, float z) {
        return make_float3(x, y, z);
    }

    __host__ __device__ static float3 add(const float3& a, const float3& b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __host__ __device__ static float3 sub(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __host__ __device__ static float3 mul(const float3& a, float t) {
        return make_float3(a.x * t, a.y * t, a.z * t);
    }

    __host__ __device__ static float dot(const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ static float length(const float3& v) {
        return sqrtf(dot(v, v));
    }

    __host__ __device__ static float3 normalize(const float3& v) {
        float len = length(v);
        return make_float3(v.x/len, v.y/len, v.z/len);
    }

    __host__ __device__ static float3 reflect(const float3& v, const float3& n) {
        return sub(v, mul(n, 2.0f * dot(v, n)));
    }

    __host__ __device__ static float3 lerp(const float3& a, const float3& b, float t) {
        return add(mul(a, 1.0f - t), mul(b, t));
    }
};

struct GPURay {
    float3 origin;
    float3 direction;
    
    __device__ float3 at(float t) const {
        return float3_ops::add(origin, float3_ops::mul(direction, t));
    }
};

// =========================================================
// GPU Sphere and Material Structures
// =========================================================

// Constants for optimization
#define MAX_LIGHTS 10
#define MAX_SPHERES_PER_BLOCK 256

struct GPUMaterial {
    float3 albedo;
    float metallic;
    float roughness;
    float shininess;
};

struct GPULight {
    float3 position;
    float3 color;
    float intensity;
};

// Constant memory for lights (read-only, cached, broadcast to all threads)
__constant__ GPULight const_lights[MAX_LIGHTS];
__constant__ int const_num_lights;
__constant__ float3 const_ambient_light;

struct GPUSphere {
    float3 center;
    float radius;
    GPUMaterial material;

    // GPU ray-sphere intersection using quadratic formula
    __device__ bool intersect(const GPURay& ray, float t_min, float t_max, float& t) const {
        // Vector from ray origin to sphere center
        float3 oc = float3_ops::sub(ray.origin, center);

        // Quadratic equation coefficients: at^2 + bt + c = 0
        float a = float3_ops::dot(ray.direction, ray.direction);
        float b = 2.0f * float3_ops::dot(oc, ray.direction);
        float c = float3_ops::dot(oc, oc) - radius * radius;

        // Calculate discriminant
        float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0.0f) {
            return false;  // No intersection
        }

        // Find the nearest intersection in range [t_min, t_max]
        float sqrt_disc = sqrtf(discriminant);
        float t1 = (-b - sqrt_disc) / (2.0f * a);
        float t2 = (-b + sqrt_disc) / (2.0f * a);

        // Try nearest intersection first
        if (t1 >= t_min && t1 <= t_max) {
            t = t1;
            return true;
        }

        // Try farther intersection
        if (t2 >= t_min && t2 <= t_max) {
            t = t2;
            return true;
        }

        return false;
    }

    __device__ float3 normal_at(const float3& point) const {
        return float3_ops::normalize(float3_ops::sub(point, center));
    }
};

// =========================================================
// GPU Camera
// =========================================================

struct GPUCamera {
    float3 origin;
    float3 lower_left;
    float3 horizontal;
    float3 vertical;
    
    __device__ GPURay get_ray(float u, float v) const {
        float3 direction = float3_ops::add(
            lower_left,
            float3_ops::add(
                float3_ops::mul(horizontal, u),
                float3_ops::mul(vertical, v)
            )
        );
        direction = float3_ops::sub(direction, origin);
        
        GPURay ray;
        ray.origin = origin;
        ray.direction = float3_ops::normalize(direction);
        return ray;
    }
};

// =========================================================
// TODO: STUDENT IMPLEMENTATION - GPU Ray Tracing Kernel
// =========================================================
// Implement the main ray tracing kernel that runs on the GPU.
// Each thread handles one pixel.
// 
// Key differences from CPU version:
// - No recursion (use iterative approach for reflections)
// - Use shared memory for frequently accessed data
// - Be careful with memory access patterns
// =========================================================

__global__ void render_kernel(float3* framebuffer,
                             GPUSphere* spheres, int num_spheres,
                             GPULight* lights, int num_lights,
                             GPUCamera camera,
                             int width, int height,
                             int max_bounces) {

    // Calculate pixel coordinates (one thread per pixel)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 1. Generate primary ray for this pixel
    float u = float(x) / float(width - 1);
    float v = float(y) / float(height - 1);
    GPURay ray = camera.get_ray(u, v);

    // 2. Initialize color accumulator and attenuation
    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);  // Tracks how much light makes it back
    float3 ambient_light = make_float3(0.1f, 0.1f, 0.12f);

    // 3. Iterative ray bouncing (replaces recursion)
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        // a. Find closest intersection
        float closest_t = INFINITY;
        int hit_sphere_idx = -1;

        for (int i = 0; i < num_spheres; i++) {
            float t;
            if (spheres[i].intersect(ray, 0.001f, closest_t, t)) {
                closest_t = t;
                hit_sphere_idx = i;
            }
        }

        // b. If no hit, add background color and break
        if (hit_sphere_idx < 0) {
            // Sky gradient background
            float t = 0.5f * (ray.direction.y + 1.0f);
            float3 sky_color = float3_ops::lerp(
                make_float3(1.0f, 1.0f, 1.0f),
                make_float3(0.5f, 0.7f, 1.0f),
                t
            );
            final_color = float3_ops::add(final_color,
                float3_ops::mul(sky_color, attenuation.x));  // Simple attenuation
            break;
        }

        // c. Calculate shading at hit point
        const GPUSphere& hit_sphere = spheres[hit_sphere_idx];
        float3 hit_point = ray.at(closest_t);
        float3 normal = hit_sphere.normal_at(hit_point);
        float3 view_dir = float3_ops::normalize(float3_ops::sub(ray.origin, hit_point));

        // Start with ambient lighting
        float3 surface_color = float3_ops::mul(
            float3_ops::mul(ambient_light, hit_sphere.material.albedo.x),
            hit_sphere.material.albedo.x
        );
        surface_color.y = ambient_light.y * hit_sphere.material.albedo.y;
        surface_color.z = ambient_light.z * hit_sphere.material.albedo.z;

        // For each light, add diffuse and specular contributions
        for (int l = 0; l < num_lights; l++) {
            const GPULight& light = lights[l];

            // Check for shadows
            float3 to_light = float3_ops::sub(light.position, hit_point);
            float light_distance = float3_ops::length(to_light);
            float3 light_dir = float3_ops::normalize(to_light);

            // Shadow ray
            GPURay shadow_ray;
            shadow_ray.origin = float3_ops::add(hit_point, float3_ops::mul(normal, 0.001f));
            shadow_ray.direction = light_dir;

            bool in_shadow = false;
            for (int s = 0; s < num_spheres; s++) {
                float shadow_t;
                if (spheres[s].intersect(shadow_ray, 0.001f, light_distance, shadow_t)) {
                    in_shadow = true;
                    break;
                }
            }

            if (!in_shadow) {
                // Diffuse (Lambert)
                float n_dot_l = fmaxf(0.0f, float3_ops::dot(normal, light_dir));
                float3 diffuse = make_float3(
                    hit_sphere.material.albedo.x * light.color.x * light.intensity * n_dot_l,
                    hit_sphere.material.albedo.y * light.color.y * light.intensity * n_dot_l,
                    hit_sphere.material.albedo.z * light.color.z * light.intensity * n_dot_l
                );

                // Specular (Phong)
                float3 reflect_dir = float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
                float r_dot_v = fmaxf(0.0f, float3_ops::dot(reflect_dir, view_dir));
                float spec_factor = powf(r_dot_v, hit_sphere.material.shininess);
                float3 specular = float3_ops::mul(
                    float3_ops::mul(light.color, light.intensity * hit_sphere.material.metallic),
                    spec_factor
                );

                surface_color = float3_ops::add(surface_color, float3_ops::add(diffuse, specular));
            }
        }

        // d. Handle reflections
        float reflectivity = hit_sphere.material.metallic;
        if (reflectivity > 0.01f && bounce < max_bounces - 1) {
            // Accumulate current surface color weighted by (1 - reflectivity)
            float3 direct_contribution = make_float3(
                surface_color.x * attenuation.x * (1.0f - reflectivity),
                surface_color.y * attenuation.y * (1.0f - reflectivity),
                surface_color.z * attenuation.z * (1.0f - reflectivity)
            );
            final_color = float3_ops::add(final_color, direct_contribution);

            // Update attenuation for reflection
            attenuation.x *= reflectivity;
            attenuation.y *= reflectivity;
            attenuation.z *= reflectivity;

            // Setup ray for next bounce
            ray.origin = float3_ops::add(hit_point, float3_ops::mul(normal, 0.001f));
            ray.direction = float3_ops::reflect(ray.direction, normal);
        } else {
            // e. Accumulate final color and stop
            float3 contribution = make_float3(
                surface_color.x * attenuation.x,
                surface_color.y * attenuation.y,
                surface_color.z * attenuation.z
            );
            final_color = float3_ops::add(final_color, contribution);
            break;
        }
    }

    // 4. Store final color in framebuffer
    int pixel_idx = y * width + x;
    framebuffer[pixel_idx] = final_color;
}

// =========================================================
// OPTIMIZED KERNEL - Shared Memory + Constant Memory
// =========================================================
// Uses shared memory for spheres and constant memory for lights
// for improved memory access patterns and performance.
// =========================================================

__global__ void render_kernel_optimized(float3* framebuffer,
                                       GPUSphere* global_spheres, int num_spheres,
                                       GPUCamera camera,
                                       int width, int height,
                                       int max_bounces) {

    // 1. Declare shared memory for spheres (dynamically allocated)
    extern __shared__ GPUSphere shared_spheres[];

    // 2. Cooperatively load spheres into shared memory
    // Each thread in the block loads one or more spheres
    int threads_per_block = blockDim.x * blockDim.y;
    int thread_idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    // Load spheres in a coalesced manner
    for (int i = thread_idx_in_block; i < num_spheres; i += threads_per_block) {
        shared_spheres[i] = global_spheres[i];
    }

    // 3. Synchronize to ensure all spheres are loaded
    __syncthreads();

    // Calculate pixel coordinates (one thread per pixel)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Generate primary ray for this pixel
    float u = float(x) / float(width - 1);
    float v = float(y) / float(height - 1);
    GPURay ray = camera.get_ray(u, v);

    // Initialize color accumulator and attenuation
    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);

    // Iterative ray bouncing (replaces recursion)
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        // 4. Find closest intersection using SHARED MEMORY
        float closest_t = INFINITY;
        int hit_sphere_idx = -1;

        for (int i = 0; i < num_spheres; i++) {
            float t;
            if (shared_spheres[i].intersect(ray, 0.001f, closest_t, t)) {
                closest_t = t;
                hit_sphere_idx = i;
            }
        }

        // If no hit, add background color and break
        if (hit_sphere_idx < 0) {
            float t = 0.5f * (ray.direction.y + 1.0f);
            float3 sky_color = float3_ops::lerp(
                make_float3(1.0f, 1.0f, 1.0f),
                make_float3(0.5f, 0.7f, 1.0f),
                t
            );
            final_color = float3_ops::add(final_color,
                float3_ops::mul(sky_color, attenuation.x));
            break;
        }

        // Calculate shading at hit point
        const GPUSphere& hit_sphere = shared_spheres[hit_sphere_idx];
        float3 hit_point = ray.at(closest_t);
        float3 normal = hit_sphere.normal_at(hit_point);
        float3 view_dir = float3_ops::normalize(float3_ops::sub(ray.origin, hit_point));

        // Start with ambient lighting from CONSTANT MEMORY
        float3 surface_color = make_float3(
            const_ambient_light.x * hit_sphere.material.albedo.x,
            const_ambient_light.y * hit_sphere.material.albedo.y,
            const_ambient_light.z * hit_sphere.material.albedo.z
        );

        // For each light (using CONSTANT MEMORY), add diffuse and specular
        for (int l = 0; l < const_num_lights; l++) {
            const GPULight& light = const_lights[l];

            // Check for shadows
            float3 to_light = float3_ops::sub(light.position, hit_point);
            float light_distance = float3_ops::length(to_light);
            float3 light_dir = float3_ops::normalize(to_light);

            // Shadow ray
            GPURay shadow_ray;
            shadow_ray.origin = float3_ops::add(hit_point, float3_ops::mul(normal, 0.001f));
            shadow_ray.direction = light_dir;

            bool in_shadow = false;
            for (int s = 0; s < num_spheres; s++) {
                float shadow_t;
                if (shared_spheres[s].intersect(shadow_ray, 0.001f, light_distance, shadow_t)) {
                    in_shadow = true;
                    break;
                }
            }

            if (!in_shadow) {
                // Diffuse (Lambert)
                float n_dot_l = fmaxf(0.0f, float3_ops::dot(normal, light_dir));
                float3 diffuse = make_float3(
                    hit_sphere.material.albedo.x * light.color.x * light.intensity * n_dot_l,
                    hit_sphere.material.albedo.y * light.color.y * light.intensity * n_dot_l,
                    hit_sphere.material.albedo.z * light.color.z * light.intensity * n_dot_l
                );

                // Specular (Phong)
                float3 reflect_dir = float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
                float r_dot_v = fmaxf(0.0f, float3_ops::dot(reflect_dir, view_dir));
                float spec_factor = powf(r_dot_v, hit_sphere.material.shininess);
                float3 specular = float3_ops::mul(
                    float3_ops::mul(light.color, light.intensity * hit_sphere.material.metallic),
                    spec_factor
                );

                surface_color = float3_ops::add(surface_color, float3_ops::add(diffuse, specular));
            }
        }

        // Handle reflections
        float reflectivity = hit_sphere.material.metallic;
        if (reflectivity > 0.01f && bounce < max_bounces - 1) {
            // Accumulate current surface color weighted by (1 - reflectivity)
            float3 direct_contribution = make_float3(
                surface_color.x * attenuation.x * (1.0f - reflectivity),
                surface_color.y * attenuation.y * (1.0f - reflectivity),
                surface_color.z * attenuation.z * (1.0f - reflectivity)
            );
            final_color = float3_ops::add(final_color, direct_contribution);

            // Update attenuation for reflection
            attenuation.x *= reflectivity;
            attenuation.y *= reflectivity;
            attenuation.z *= reflectivity;

            // Setup ray for next bounce
            ray.origin = float3_ops::add(hit_point, float3_ops::mul(normal, 0.001f));
            ray.direction = float3_ops::reflect(ray.direction, normal);
        } else {
            // Accumulate final color and stop
            float3 contribution = make_float3(
                surface_color.x * attenuation.x,
                surface_color.y * attenuation.y,
                surface_color.z * attenuation.z
            );
            final_color = float3_ops::add(final_color, contribution);
            break;
        }
    }

    // Store final color in framebuffer
    int pixel_idx = y * width + x;
    framebuffer[pixel_idx] = final_color;
}

// =========================================================
// ANTI-ALIASING KERNEL - Multiple Samples per Pixel
// =========================================================
// Uses cuRAND for random sampling within each pixel
// =========================================================

__global__ void render_kernel_aa(float3* framebuffer,
                                  GPUSphere* global_spheres, int num_spheres,
                                  GPUCamera camera,
                                  int width, int height,
                                  int max_bounces,
                                  int samples_per_pixel,
                                  unsigned int seed) {

    // Declare shared memory for spheres (dynamically allocated)
    extern __shared__ GPUSphere shared_spheres[];

    // Cooperatively load spheres into shared memory
    int threads_per_block = blockDim.x * blockDim.y;
    int thread_idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = thread_idx_in_block; i < num_spheres; i += threads_per_block) {
        shared_spheres[i] = global_spheres[i];
    }

    __syncthreads();

    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Initialize random number generator for this pixel
    curandState rand_state;
    curand_init(seed + y * width + x, 0, 0, &rand_state);

    // Accumulate color from multiple samples
    float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);

    for (int sample = 0; sample < samples_per_pixel; sample++) {
        // Add random offset within pixel for anti-aliasing
        float rand_x = curand_uniform(&rand_state);
        float rand_y = curand_uniform(&rand_state);
        float u = float(x + rand_x) / float(width - 1);
        float v = float(y + rand_y) / float(height - 1);

        GPURay ray = camera.get_ray(u, v);
        float3 sample_color = make_float3(0.0f, 0.0f, 0.0f);
        float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);

        // Ray bouncing loop
        for (int bounce = 0; bounce < max_bounces; bounce++) {
            float closest_t = INFINITY;
            int hit_sphere_idx = -1;

            for (int i = 0; i < num_spheres; i++) {
                float t;
                if (shared_spheres[i].intersect(ray, 0.001f, closest_t, t)) {
                    closest_t = t;
                    hit_sphere_idx = i;
                }
            }

            if (hit_sphere_idx < 0) {
                float t = 0.5f * (ray.direction.y + 1.0f);
                float3 sky_color = float3_ops::lerp(
                    make_float3(1.0f, 1.0f, 1.0f),
                    make_float3(0.5f, 0.7f, 1.0f),
                    t
                );
                sample_color = float3_ops::add(sample_color,
                    float3_ops::mul(sky_color, attenuation.x));
                break;
            }

            const GPUSphere& hit_sphere = shared_spheres[hit_sphere_idx];
            float3 hit_point = ray.at(closest_t);
            float3 normal = hit_sphere.normal_at(hit_point);
            float3 view_dir = float3_ops::normalize(float3_ops::sub(ray.origin, hit_point));

            float3 surface_color = make_float3(
                const_ambient_light.x * hit_sphere.material.albedo.x,
                const_ambient_light.y * hit_sphere.material.albedo.y,
                const_ambient_light.z * hit_sphere.material.albedo.z
            );

            for (int l = 0; l < const_num_lights; l++) {
                const GPULight& light = const_lights[l];
                float3 to_light = float3_ops::sub(light.position, hit_point);
                float light_distance = float3_ops::length(to_light);
                float3 light_dir = float3_ops::normalize(to_light);

                GPURay shadow_ray;
                shadow_ray.origin = float3_ops::add(hit_point, float3_ops::mul(normal, 0.001f));
                shadow_ray.direction = light_dir;

                bool in_shadow = false;
                for (int s = 0; s < num_spheres; s++) {
                    float shadow_t;
                    if (shared_spheres[s].intersect(shadow_ray, 0.001f, light_distance, shadow_t)) {
                        in_shadow = true;
                        break;
                    }
                }

                if (!in_shadow) {
                    float n_dot_l = fmaxf(0.0f, float3_ops::dot(normal, light_dir));
                    float3 diffuse = make_float3(
                        hit_sphere.material.albedo.x * light.color.x * light.intensity * n_dot_l,
                        hit_sphere.material.albedo.y * light.color.y * light.intensity * n_dot_l,
                        hit_sphere.material.albedo.z * light.color.z * light.intensity * n_dot_l
                    );

                    float3 reflect_dir = float3_ops::reflect(float3_ops::mul(light_dir, -1.0f), normal);
                    float r_dot_v = fmaxf(0.0f, float3_ops::dot(reflect_dir, view_dir));
                    float spec_factor = powf(r_dot_v, hit_sphere.material.shininess);
                    float3 specular = float3_ops::mul(
                        float3_ops::mul(light.color, light.intensity * hit_sphere.material.metallic),
                        spec_factor
                    );

                    surface_color = float3_ops::add(surface_color, float3_ops::add(diffuse, specular));
                }
            }

            float reflectivity = hit_sphere.material.metallic;
            if (reflectivity > 0.01f && bounce < max_bounces - 1) {
                float3 direct_contribution = make_float3(
                    surface_color.x * attenuation.x * (1.0f - reflectivity),
                    surface_color.y * attenuation.y * (1.0f - reflectivity),
                    surface_color.z * attenuation.z * (1.0f - reflectivity)
                );
                sample_color = float3_ops::add(sample_color, direct_contribution);

                attenuation.x *= reflectivity;
                attenuation.y *= reflectivity;
                attenuation.z *= reflectivity;

                ray.origin = float3_ops::add(hit_point, float3_ops::mul(normal, 0.001f));
                ray.direction = float3_ops::reflect(ray.direction, normal);
            } else {
                float3 contribution = make_float3(
                    surface_color.x * attenuation.x,
                    surface_color.y * attenuation.y,
                    surface_color.z * attenuation.z
                );
                sample_color = float3_ops::add(sample_color, contribution);
                break;
            }
        }

        // Accumulate sample
        pixel_color = float3_ops::add(pixel_color, sample_color);
    }

    // Average all samples
    pixel_color.x /= float(samples_per_pixel);
    pixel_color.y /= float(samples_per_pixel);
    pixel_color.z /= float(samples_per_pixel);

    // Store final color in framebuffer
    int pixel_idx = y * width + x;
    framebuffer[pixel_idx] = pixel_color;
}

// =========================================================
// Host Functions
// =========================================================

void write_ppm(const std::string& filename, const std::vector<float3>& framebuffer,
               int width, int height) {
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";

    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            float3 color = framebuffer[j * width + i];
            int r = int(255.99f * fminf(1.0f, color.x));
            int g = int(255.99f * fminf(1.0f, color.y));
            int b = int(255.99f * fminf(1.0f, color.z));
            file << r << " " << g << " " << b << "\n";
        }
    }
}

GPUCamera setup_camera(int width, int height) {
    // Camera parameters
    float3 lookfrom = make_float3(0, 2, 5);
    float3 lookat = make_float3(0, 0, -20);
    float3 vup = make_float3(0, 1, 0);
    float vfov = 60.0f;
    float aspect = float(width) / float(height);
    
    // Calculate camera basis
    float theta = vfov * M_PI / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect * viewport_height;
    float focal_length = 1.0f;
    
    float3 w = float3_ops::normalize(float3_ops::sub(lookfrom, lookat));
    float3 u = float3_ops::normalize(make_float3(
        vup.y * w.z - vup.z * w.y,
        vup.z * w.x - vup.x * w.z,
        vup.x * w.y - vup.y * w.x
    ));
    float3 v = make_float3(
        w.y * u.z - w.z * u.y,
        w.z * u.x - w.x * u.z,
        w.x * u.y - w.y * u.x
    );
    
    GPUCamera camera;
    camera.origin = lookfrom;
    camera.horizontal = float3_ops::mul(u, viewport_width);
    camera.vertical = float3_ops::mul(v, viewport_height);
    camera.lower_left = float3_ops::sub(
        float3_ops::sub(
            float3_ops::sub(lookfrom, float3_ops::mul(camera.horizontal, 0.5f)),
            float3_ops::mul(camera.vertical, 0.5f)
        ),
        float3_ops::mul(w, focal_length)
    );
    
    return camera;
}

// Load scene from file
bool load_scene_gpu(const std::string& filename,
                    std::vector<GPUSphere>& spheres,
                    std::vector<GPULight>& lights,
                    float3& ambient_light,
                    GPUCamera& camera,
                    int width, int height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open scene file: " << filename << std::endl;
        return false;
    }

    std::string line;
    float3 cam_pos = make_float3(0, 0, 0);
    float3 cam_lookat = make_float3(0, 0, -1);
    float cam_fov = 60.0f;
    ambient_light = make_float3(0.1f, 0.1f, 0.1f);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "sphere") {
            float x, y, z, radius, r, g, b, metallic, roughness, shininess;
            if (iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >> roughness >> shininess) {
                GPUSphere sphere;
                sphere.center = make_float3(x, y, z);
                sphere.radius = radius;
                sphere.material.albedo = make_float3(r, g, b);
                sphere.material.metallic = metallic;
                sphere.material.roughness = roughness;
                sphere.material.shininess = shininess;
                spheres.push_back(sphere);
            }
        } else if (type == "light") {
            float x, y, z, r, g, b, intensity;
            if (iss >> x >> y >> z >> r >> g >> b >> intensity) {
                GPULight light;
                light.position = make_float3(x, y, z);
                light.color = make_float3(r, g, b);
                light.intensity = intensity;
                lights.push_back(light);
            }
        } else if (type == "ambient") {
            float r, g, b;
            if (iss >> r >> g >> b) {
                ambient_light = make_float3(r, g, b);
            }
        } else if (type == "camera") {
            float px, py, pz, lx, ly, lz, fov;
            if (iss >> px >> py >> pz >> lx >> ly >> lz >> fov) {
                cam_pos = make_float3(px, py, pz);
                cam_lookat = make_float3(lx, ly, lz);
                cam_fov = fov;
            }
        }
    }

    file.close();

    // Setup camera from loaded parameters
    float3 vup = make_float3(0, 1, 0);
    float aspect = float(width) / float(height);
    float theta = cam_fov * M_PI / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect * viewport_height;
    float focal_length = 1.0f;

    float3 w = float3_ops::normalize(float3_ops::sub(cam_pos, cam_lookat));
    float3 u = float3_ops::normalize(make_float3(
        vup.y * w.z - vup.z * w.y,
        vup.z * w.x - vup.x * w.z,
        vup.x * w.y - vup.y * w.x
    ));
    float3 v = make_float3(
        w.y * u.z - w.z * u.y,
        w.z * u.x - w.x * u.z,
        w.x * u.y - w.y * u.x
    );

    camera.origin = cam_pos;
    camera.horizontal = float3_ops::mul(u, viewport_width);
    camera.vertical = float3_ops::mul(v, viewport_height);
    camera.lower_left = float3_ops::sub(
        float3_ops::sub(
            float3_ops::sub(cam_pos, float3_ops::mul(camera.horizontal, 0.5f)),
            float3_ops::mul(camera.vertical, 0.5f)
        ),
        float3_ops::mul(w, focal_length)
    );

    std::cout << "Loaded scene from " << filename << ":\n";
    std::cout << "  Spheres: " << spheres.size() << "\n";
    std::cout << "  Lights: " << lights.size() << "\n";

    return true;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string scene_file = "scenes/complex.txt";  // Default to complex scene
    bool use_antialiasing = false;
    int samples_per_pixel = 4;

    if (argc > 1) {
        scene_file = argv[1];
    }
    if (argc > 2) {
        use_antialiasing = (std::string(argv[2]) == "aa");
        if (argc > 3) {
            samples_per_pixel = std::atoi(argv[3]);
        }
    }

    std::cout << "Usage: " << argv[0] << " [scene_file] [aa] [samples_per_pixel]\n";
    std::cout << "Using scene: " << scene_file << "\n";
    if (use_antialiasing) {
        std::cout << "Anti-aliasing enabled with " << samples_per_pixel << " samples per pixel\n";
    }
    std::cout << "\n";

    // Image settings
    const int width = 800;
    const int height = 600;
    const int max_bounces = 3;

    // CUDA device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "Using GPU: " << props.name << std::endl;
    std::cout << "  SM Count: " << props.multiProcessorCount << std::endl;
    std::cout << "  Shared Memory per Block: " << props.sharedMemPerBlock << " bytes\n";
    std::cout << "  Max Threads per Block: " << props.maxThreadsPerBlock << "\n\n";

    // Create scene data
    std::vector<GPUSphere> h_spheres;
    std::vector<GPULight> h_lights;
    float3 h_ambient_light;
    GPUCamera camera;

    // Load scene from file
    if (!load_scene_gpu(scene_file, h_spheres, h_lights, h_ambient_light, camera, width, height)) {
        std::cerr << "Failed to load scene. Exiting.\n";
        return 1;
    }

    // Check limits
    if (h_lights.size() > MAX_LIGHTS) {
        std::cerr << "Warning: Scene has " << h_lights.size() << " lights, but MAX_LIGHTS is "
                  << MAX_LIGHTS << ". Using first " << MAX_LIGHTS << " lights only.\n";
        h_lights.resize(MAX_LIGHTS);
    }

    std::cout << "Scene loaded: " << h_spheres.size() << " spheres, "
              << h_lights.size() << " lights\n\n";

    // Copy lights and ambient to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(const_lights, h_lights.data(),
                                  h_lights.size() * sizeof(GPULight)));
    int num_lights = h_lights.size();
    CUDA_CHECK(cudaMemcpyToSymbol(const_num_lights, &num_lights, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_ambient_light, &h_ambient_light, sizeof(float3)));
    std::cout << "Copied " << num_lights << " lights to constant memory\n";

    // Allocate device memory for spheres and framebuffer
    GPUSphere* d_spheres;
    float3* d_framebuffer;

    CUDA_CHECK(cudaMalloc(&d_spheres, h_spheres.size() * sizeof(GPUSphere)));
    CUDA_CHECK(cudaMalloc(&d_framebuffer, width * height * sizeof(float3)));

    // Copy spheres to device global memory
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(),
                          h_spheres.size() * sizeof(GPUSphere),
                          cudaMemcpyHostToDevice));
    std::cout << "Copied " << h_spheres.size() << " spheres to global memory\n";

    // Configure kernel launch
    // Use 16x16 threads per block (256 threads, multiple of 32 for warp efficiency)
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    std::cout << "\nKernel Configuration:\n";
    std::cout << "  Grid: " << blocks.x << " x " << blocks.y << " blocks\n";
    std::cout << "  Block: " << threads.x << " x " << threads.y << " threads\n";
    std::cout << "  Total threads: " << (blocks.x * blocks.y * threads.x * threads.y) << "\n";

    // Calculate shared memory size for spheres
    size_t shared_size = h_spheres.size() * sizeof(GPUSphere);
    std::cout << "  Shared memory per block: " << shared_size << " bytes\n";

    if (shared_size > props.sharedMemPerBlock) {
        std::cerr << "WARNING: Shared memory required (" << shared_size
                  << " bytes) exceeds available (" << props.sharedMemPerBlock
                  << " bytes). Kernel may fail.\n";
    }

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Render
    std::cout << "\nRendering..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));

    if (use_antialiasing) {
        // Use anti-aliasing kernel
        unsigned int seed = 1234;
        render_kernel_aa<<<blocks, threads, shared_size>>>(
            d_framebuffer, d_spheres, h_spheres.size(),
            camera, width, height, max_bounces,
            samples_per_pixel, seed
        );
    } else {
        // Use optimized kernel with shared memory
        render_kernel_optimized<<<blocks, threads, shared_size>>>(
            d_framebuffer, d_spheres, h_spheres.size(),
            camera, width, height, max_bounces
        );
    }

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU rendering time: " << milliseconds / 1000.0f << " seconds\n";

    // Copy result back to host
    std::vector<float3> h_framebuffer(width * height);
    CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), d_framebuffer,
                          width * height * sizeof(float3),
                          cudaMemcpyDeviceToHost));

    // Write output
    std::string output_file = use_antialiasing ? "output_gpu_aa.ppm" : "output_gpu.ppm";
    write_ppm(output_file, h_framebuffer, width, height);

    // Cleanup
    CUDA_CHECK(cudaFree(d_spheres));
    CUDA_CHECK(cudaFree(d_framebuffer));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Done! Output written to " << output_file << "\n";

    return 0;
}
