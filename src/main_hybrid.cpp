// main_hybrid.cpp - Week 3: Hybrid CPU-GPU Ray Tracer
// CS420 Ray Tracer Project
// Status: TEMPLATE - STUDENT MUST COMPLETE

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <queue>
#include <thread>
#include <atomic>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "math_constants.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"

// CUDA runtime API (for hybrid execution)
#include <cuda_runtime.h>

// =========================================================
// Image Output Functions
// =========================================================

void write_ppm(const std::string &filename, const std::vector<Vec3> &framebuffer,
               int width, int height)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // PPM header
    file << "P3\n"
         << width << " " << height << "\n255\n";

    // Write pixels (PPM is top-to-bottom)
    for (int j = height - 1; j >= 0; j--)
    {
        for (int i = 0; i < width; i++)
        {
            int r, g, b;
            framebuffer[j * width + i].to_rgb(r, g, b);
            file << r << " " << g << " " << b << "\n";
        }
    }

    file.close();
    std::cout << "Image written to " << filename << std::endl;
}

// =========================================================
// Scene Creation (reuse from Week 1)
// =========================================================
Scene create_test_scene()
{
    Scene scene;

    // Add spheres
    scene.add_sphere(Sphere(Vec3(0, 0, -20), 2.0,
                            Material::diffuse(Vec3(1, 0, 0))));
    scene.add_sphere(Sphere(Vec3(3, 0, -20), 2.0,
                            Material::metal(Vec3(0.8, 0.8, 0.8), 0.1)));
    scene.add_sphere(Sphere(Vec3(-3, 0, -20), 2.0,
                            Material::diffuse(Vec3(0, 0, 1))));
    scene.add_sphere(Sphere(Vec3(0, -102, -20), 100.0,
                            Material::diffuse(Vec3(0.5, 0.5, 0.5))));

    // Add lights
    scene.add_light(Light(Vec3(10, 10, -10), Vec3(1, 1, 1), 0.7));
    scene.add_light(Light(Vec3(-10, 10, -10), Vec3(1, 1, 0.8), 0.5));

    // Set ambient
    scene.set_ambient(Vec3(0.1, 0.1, 0.1));

    return scene;
}

// =========================================================
// Tile Structure for Work Distribution
// =========================================================
struct Tile
{
    int x_start, y_start;
    int x_end, y_end;
    int complexity_estimate; // Estimated work for this tile
    bool processed;

    Tile(int xs, int ys, int xe, int ye)
        : x_start(xs), y_start(ys), x_end(xe), y_end(ye),
          complexity_estimate(0), processed(false) {}

    int pixel_count() const
    {
        return (x_end - x_start) * (y_end - y_start);
    }
};

// =========================================================
// GPU Kernel Declaration (implemented in kernel.cu)
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
    cudaStream_t stream);

// =========================================================
// CPU Ray Tracing (Complex Shading Path)
// =========================================================
Vec3 trace_ray_cpu(const Ray &ray, const Scene &scene, int depth)
{
    // TODO: STUDENT - Implement CPU ray tracing
    // This should handle complex shading, deep reflections, etc.
    // Can reuse code from Week 1

    return scene.get_background(ray);
}

void process_tile_cpu(const Tile &tile, const Scene &scene, const Camera &camera,
                      std::vector<Vec3> &framebuffer, int width, int height, int max_depth)
{

#pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int y = tile.y_start; y < tile.y_end; y++)
    {
        for (int x = tile.x_start; x < tile.x_end; x++)
        {
            Ray ray = camera.get_ray_pixel(x, y, width, height);
            framebuffer[y * width + x] = trace_ray_cpu(ray, scene, max_depth);
        }
    }
}

// =========================================================
// GPU Memory Management
// =========================================================
class GPUResources
{
private:
    float *d_framebuffer;
    float *d_spheres;
    float *d_lights;
    size_t fb_size;
    size_t spheres_size;
    size_t lights_size;

public:
    GPUResources(int width, int height, int num_spheres, int num_lights)
    {
        fb_size = width * height * 3 * sizeof(float);
        spheres_size = num_spheres * 8 * sizeof(float); // center(3) + radius(1) + material(4)
        lights_size = num_lights * 7 * sizeof(float);   // position(3) + color(3) + intensity(1)

        cudaMalloc(&d_framebuffer, fb_size);
        cudaMalloc(&d_spheres, spheres_size);
        cudaMalloc(&d_lights, lights_size);
    }

    ~GPUResources()
    {
        cudaFree(d_framebuffer);
        cudaFree(d_spheres);
        cudaFree(d_lights);
    }

    void upload_scene(const Scene &scene)
    {
        // TODO: STUDENT - Convert scene data to GPU format and upload
        // Pack spheres and lights into float arrays
    }

    void download_tile(const Tile &tile, std::vector<Vec3> &framebuffer,
                       int width, cudaStream_t stream)
    {
        // TODO: STUDENT - Download tile results from GPU to CPU framebuffer
    }

    float *get_framebuffer() { return d_framebuffer; }
    float *get_spheres() { return d_spheres; }
    float *get_lights() { return d_lights; }
};

// =========================================================
// Tile Complexity Estimation
// =========================================================
int estimate_tile_complexity(const Tile &tile, const Scene &scene, const Camera &camera)
{
    // TODO: STUDENT - Implement heuristic to estimate rendering complexity
    // Consider:
    // - Number of spheres likely to be intersected
    // - Presence of reflective materials
    // - Distance from camera
    // Simple version: sample a few rays and count intersections

    return tile.pixel_count(); // Placeholder: just use pixel count
}

// =========================================================
// TODO: STUDENT IMPLEMENTATION - Hybrid Work Distribution
// =========================================================
// Design and implement the work distribution strategy.
// Decide which tiles go to CPU vs GPU based on:
// - Complexity estimates
// - Current workload
// - Memory constraints
// Use CUDA streams for overlapping computation
// =========================================================

void render_hybrid(const Scene &scene, const Camera &camera,
                   std::vector<Vec3> &framebuffer,
                   int width, int height, int max_depth,
                   int tile_size = 64)
{

    std::cout << "Hybrid Rendering..." << std::endl;

    // Create tiles
    std::vector<Tile> tiles;
    for (int y = 0; y < height; y += tile_size)
    {
        for (int x = 0; x < width; x += tile_size)
        {
            int xe = std::min(x + tile_size, width);
            int ye = std::min(y + tile_size, height);
            tiles.emplace_back(x, y, xe, ye);
        }
    }

    std::cout << "Created " << tiles.size() << " tiles of size "
              << tile_size << "x" << tile_size << std::endl;

    // Estimate complexity for each tile
    for (auto &tile : tiles)
    {
        tile.complexity_estimate = estimate_tile_complexity(tile, scene, camera);
    }

    // Initialize GPU resources
    GPUResources gpu_resources(width, height,
                               scene.get_spheres().size(),
                               scene.get_lights().size());
    gpu_resources.upload_scene(scene);

    // Create CUDA streams for pipelining
    const int NUM_STREAMS = 3;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // TODO: STUDENT - Implement work distribution
    // Split tiles between CPU and GPU based on complexity
    std::queue<Tile *> cpu_queue;
    std::queue<Tile *> gpu_queue;

    // Simple strategy: complex tiles to CPU, simple to GPU
    int complexity_threshold = width * height / (tile_size * tile_size) * 2;

    for (auto &tile : tiles)
    {
        if (tile.complexity_estimate > complexity_threshold)
        {
            cpu_queue.push(&tile);
        }
        else
        {
            gpu_queue.push(&tile);
        }
    }

    std::cout << "Distribution: " << cpu_queue.size() << " tiles to CPU, "
              << gpu_queue.size() << " tiles to GPU" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

// TODO: STUDENT - Process tiles concurrently
// Use OpenMP sections or std::thread for CPU-GPU concurrency
#pragma omp parallel sections
    {
// CPU processing section
#pragma omp section
        {
            while (!cpu_queue.empty())
            {
                Tile *tile = cpu_queue.front();
                cpu_queue.pop();
                process_tile_cpu(*tile, scene, camera, framebuffer, width, height, max_depth);
                tile->processed = true;
            }
        }

// GPU processing section
#pragma omp section
        {
            int stream_idx = 0;
            while (!gpu_queue.empty())
            {
                Tile *tile = gpu_queue.front();
                gpu_queue.pop();

                // TODO: STUDENT - Launch GPU kernel for this tile
                // Use streams for asynchronous execution
                cudaStream_t stream = streams[stream_idx];
                stream_idx = (stream_idx + 1) % NUM_STREAMS;

                // launch_gpu_kernel(..., stream);

                tile->processed = true;
            }

            // Wait for all GPU work to complete
            for (int i = 0; i < NUM_STREAMS; i++)
            {
                cudaStreamSynchronize(streams[i]);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Hybrid rendering time: " << diff.count() << " seconds" << std::endl;

    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    // Verify all tiles processed
    int unprocessed = 0;
    for (const auto &tile : tiles)
    {
        if (!tile.processed)
            unprocessed++;
    }
    if (unprocessed > 0)
    {
        std::cerr << "Warning: " << unprocessed << " tiles not processed!" << std::endl;
    }
}

// =========================================================
// Asynchronous Pipeline Version (Advanced)
// =========================================================
void render_hybrid_pipeline(const Scene &scene, const Camera &camera,
                            std::vector<Vec3> &framebuffer,
                            int width, int height, int max_depth)
{

    std::cout << "Hybrid Pipeline Rendering..." << std::endl;

    // TODO: STUDENT - Implement pipelined version
    // Stage 1: Tile generation and complexity estimation
    // Stage 2: GPU kernel execution
    // Stage 3: CPU processing
    // Stage 4: Result aggregation
    // Use pinned memory for faster transfers

    // Placeholder - calls basic hybrid version
    render_hybrid(scene, camera, framebuffer, width, height, max_depth);
}

// =========================================================
// Main Function
// =========================================================
int main(int argc, char *argv[])
{
    // Image settings
    const int width = 1280;
    const int height = 720;
    const int max_depth = 3;
    const double aspect_ratio = double(width) / double(height);

    // Parse command line arguments
    bool use_pipeline = false;
    int tile_size = 64;
    std::string output_file = "output_hybrid.ppm";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--pipeline" || arg == "-p")
        {
            use_pipeline = true;
        }
        else if (arg == "--tile-size" || arg == "-t")
        {
            if (i + 1 < argc)
            {
                tile_size = std::atoi(argv[++i]);
            }
        }
        else if (arg == "--output" || arg == "-o")
        {
            if (i + 1 < argc)
            {
                output_file = argv[++i];
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --pipeline, -p        Use pipelined execution\n";
            std::cout << "  --tile-size, -t SIZE  Set tile size (default: 64)\n";
            std::cout << "  --output, -o FILE     Output filename\n";
            std::cout << "  --help, -h            Show this help message\n";
            return 0;
        }
    }

    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found. Cannot run hybrid version." << std::endl;
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "Using GPU: " << props.name << std::endl;
    std::cout << "Tile size: " << tile_size << "x" << tile_size << std::endl;

    // Create scene (can load from file or create programmatically)
    Scene scene = create_test_scene(); // Reuse from Week 1
    scene.print_stats();

    // Setup camera
    Vec3 lookfrom(0, 2, 5);
    Vec3 lookat(0, 0, -20);
    Vec3 vup(0, 1, 0);
    double vfov = 60.0;

    Camera camera(lookfrom, lookat, vup, vfov, aspect_ratio);

    // Allocate framebuffer
    std::vector<Vec3> framebuffer(width * height);

    // Render
    if (use_pipeline)
    {
        render_hybrid_pipeline(scene, camera, framebuffer, width, height, max_depth);
    }
    else
    {
        render_hybrid(scene, camera, framebuffer, width, height, max_depth, tile_size);
    }

    // Write output
    write_ppm(output_file, framebuffer, width, height);

// Performance comparison
#ifdef COMPARE_MODES
    std::cout << "\n=== Performance Comparison ===" << std::endl;

    // Run CPU-only with OpenMP
    auto start = std::chrono::high_resolution_clock::now();
    // ... CPU rendering ...
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end - start).count();

    // Run GPU-only
    start = std::chrono::high_resolution_clock::now();
    // ... GPU rendering ...
    end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(end - start).count();

    // Run Hybrid
    start = std::chrono::high_resolution_clock::now();
    render_hybrid(scene, camera, framebuffer, width, height, max_depth, tile_size);
    end = std::chrono::high_resolution_clock::now();
    double hybrid_time = std::chrono::duration<double>(end - start).count();

    std::cout << "CPU-only time:    " << cpu_time << " seconds" << std::endl;
    std::cout << "GPU-only time:    " << gpu_time << " seconds" << std::endl;
    std::cout << "Hybrid time:      " << hybrid_time << " seconds" << std::endl;
    std::cout << "Hybrid speedup over GPU: " << gpu_time / hybrid_time << "x" << std::endl;
#endif

    return 0;
}
