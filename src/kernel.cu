// kernel.cu - GPU Kernels for Hybrid Ray Tracer
// CS420 Ray Tracer Project - Week 3

#include "cuda_fix.h"
#include <cuda_runtime.h>
#include <cmath>

// =========================================================
// GPU Vector Operations
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

    __host__ __device__ static float3 mul_componentwise(const float3& a, const float3& b) {
        return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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

// =========================================================
// GPU Ray Structure
// =========================================================

struct GPURay {
    float3 origin;
    float3 direction;

    __device__ float3 at(float t) const {
        return float3_ops::add(origin, float3_ops::mul(direction, t));
    }
};

// =========================================================
// GPU Material and Geometry Structures
// =========================================================

#define MAX_LIGHTS 10

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

struct GPUSphere {
    float3 center;
    float radius;
    GPUMaterial material;

    __device__ bool intersect(const GPURay& ray, float t_min, float t_max, float& t) const {
        float3 oc = float3_ops::sub(ray.origin, center);
        float a = float3_ops::dot(ray.direction, ray.direction);
        float b = 2.0f * float3_ops::dot(oc, ray.direction);
        float c = float3_ops::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0.0f) return false;

        float sqrt_disc = sqrtf(discriminant);
        float t1 = (-b - sqrt_disc) / (2.0f * a);
        float t2 = (-b + sqrt_disc) / (2.0f * a);

        if (t1 >= t_min && t1 <= t_max) {
            t = t1;
            return true;
        }
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

// Constant memory for lights
__constant__ GPULight const_lights[MAX_LIGHTS];
__constant__ int const_num_lights;
__constant__ float3 const_ambient_light;

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
// Tile-based Rendering Kernel for Hybrid System
// =========================================================
// This kernel renders a specific tile of the image.
// Multiple tiles can be processed concurrently using streams.
// =========================================================

__global__ void render_tile_kernel(
    float3* framebuffer,
    GPUSphere* spheres, int num_spheres,
    GPUCamera camera,
    int tile_x, int tile_y,
    int tile_width, int tile_height,
    int image_width, int image_height,
    int max_depth)
{
    // Calculate pixel coordinates within the tile
    int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    int local_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_x >= tile_width || local_y >= tile_height) return;

    // Convert to global image coordinates
    int x = tile_x + local_x;
    int y = tile_y + local_y;

    if (x >= image_width || y >= image_height) return;

    // Generate primary ray
    float u = float(x) / float(image_width - 1);
    float v = float(y) / float(image_height - 1);
    GPURay ray = camera.get_ray(u, v);

    // Initialize color accumulator
    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);

    // Iterative ray bouncing
    for (int bounce = 0; bounce < max_depth; bounce++) {
        float closest_t = INFINITY;
        int hit_sphere_idx = -1;

        // Find closest intersection
        for (int i = 0; i < num_spheres; i++) {
            float t;
            if (spheres[i].intersect(ray, 0.001f, closest_t, t)) {
                closest_t = t;
                hit_sphere_idx = i;
            }
        }

        // No hit - background
        if (hit_sphere_idx < 0) {
            float t = 0.5f * (ray.direction.y + 1.0f);
            float3 sky_color = float3_ops::lerp(
                make_float3(1.0f, 1.0f, 1.0f),
                make_float3(0.5f, 0.7f, 1.0f),
                t
            );
            final_color = float3_ops::add(final_color,
                float3_ops::mul_componentwise(sky_color, attenuation));
            break;
        }

        // Calculate shading
        const GPUSphere& hit_sphere = spheres[hit_sphere_idx];
        float3 hit_point = ray.at(closest_t);
        float3 normal = hit_sphere.normal_at(hit_point);
        float3 view_dir = float3_ops::normalize(float3_ops::sub(ray.origin, hit_point));

        // Ambient lighting
        float3 surface_color = float3_ops::mul_componentwise(
            const_ambient_light,
            hit_sphere.material.albedo
        );

        // Process each light
        for (int l = 0; l < const_num_lights; l++) {
            const GPULight& light = const_lights[l];
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
                // Diffuse
                float n_dot_l = fmaxf(0.0f, float3_ops::dot(normal, light_dir));
                float3 diffuse = float3_ops::mul(
                    float3_ops::mul_componentwise(hit_sphere.material.albedo, light.color),
                    light.intensity * n_dot_l
                );

                // Specular
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
        if (reflectivity > 0.01f && bounce < max_depth - 1) {
            float3 direct_contribution = float3_ops::mul_componentwise(
                surface_color,
                float3_ops::mul(attenuation, 1.0f - reflectivity)
            );
            final_color = float3_ops::add(final_color, direct_contribution);

            attenuation = float3_ops::mul(attenuation, reflectivity);

            ray.origin = float3_ops::add(hit_point, float3_ops::mul(normal, 0.001f));
            ray.direction = float3_ops::reflect(ray.direction, normal);
        } else {
            float3 contribution = float3_ops::mul_componentwise(surface_color, attenuation);
            final_color = float3_ops::add(final_color, contribution);
            break;
        }
    }

    // Store result in framebuffer
    int pixel_idx = y * image_width + x;
    framebuffer[pixel_idx] = final_color;
}

// =========================================================
// Host-callable Kernel Launcher
// =========================================================

extern "C" void launch_gpu_kernel(
    float *d_framebuffer,
    float *d_spheres, int num_spheres,
    float *d_lights, int num_lights,
    float *camera_params,
    int tile_x, int tile_y,
    int tile_width, int tile_height,
    int image_width, int image_height,
    int max_depth,
    cudaStream_t stream)
{
    // Reinterpret packed arrays as structures
    float3* fb = reinterpret_cast<float3*>(d_framebuffer);
    GPUSphere* spheres = reinterpret_cast<GPUSphere*>(d_spheres);

    // Setup camera from params
    GPUCamera camera;
    camera.origin = make_float3(camera_params[0], camera_params[1], camera_params[2]);
    camera.lower_left = make_float3(camera_params[3], camera_params[4], camera_params[5]);
    camera.horizontal = make_float3(camera_params[6], camera_params[7], camera_params[8]);
    camera.vertical = make_float3(camera_params[9], camera_params[10], camera_params[11]);

    // Configure kernel launch for tile
    dim3 threads(16, 16);
    dim3 blocks((tile_width + threads.x - 1) / threads.x,
                (tile_height + threads.y - 1) / threads.y);

    // Launch kernel
    render_tile_kernel<<<blocks, threads, 0, stream>>>(
        fb, spheres, num_spheres,
        camera,
        tile_x, tile_y, tile_width, tile_height,
        image_width, image_height, max_depth
    );
}

// Helper function to copy lights to constant memory
extern "C" void upload_lights_to_constant(float *h_lights, int num_lights, float *h_ambient) {
    GPULight lights[MAX_LIGHTS];

    // Unpack light data
    for (int i = 0; i < num_lights && i < MAX_LIGHTS; i++) {
        int idx = i * 7;
        lights[i].position = make_float3(h_lights[idx], h_lights[idx+1], h_lights[idx+2]);
        lights[i].color = make_float3(h_lights[idx+3], h_lights[idx+4], h_lights[idx+5]);
        lights[i].intensity = h_lights[idx+6];
    }

    float3 ambient = make_float3(h_ambient[0], h_ambient[1], h_ambient[2]);

    cudaMemcpyToSymbol(const_lights, lights, num_lights * sizeof(GPULight));
    cudaMemcpyToSymbol(const_num_lights, &num_lights, sizeof(int));
    cudaMemcpyToSymbol(const_ambient_light, &ambient, sizeof(float3));
}
