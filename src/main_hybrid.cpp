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
            Vec3 color = framebuffer[j * width + i];
            int r = int(255.99 * std::min(1.0, color.x));
            int g = int(255.99 * std::min(1.0, color.y));
            int b = int(255.99 * std::min(1.0, color.z));
            file << r << " " << g << " " << b << "\n";
        }
    }

    file.close();
    std::cout << "Image written to " << filename << std::endl;
}

// =========================================================
// Scene Creation and Loading
// =========================================================
Scene create_test_scene()
{
    Scene scene;

    // Add spheres
    add_sphere(scene, Sphere(Vec3(0, 0, -20), 2.0,
                            Material{Vec3(1, 0, 0), 0.0, 32.0}));
    add_sphere(scene, Sphere(Vec3(3, 0, -20), 2.0,
                            Material{Vec3(0.8, 0.8, 0.8), 0.8, 64.0}));
    add_sphere(scene, Sphere(Vec3(-3, 0, -20), 2.0,
                            Material{Vec3(0, 0, 1), 0.0, 32.0}));
    add_sphere(scene, Sphere(Vec3(0, -102, -20), 100.0,
                            Material{Vec3(0.5, 0.5, 0.5), 0.0, 32.0}));

    // Add lights
    add_light(scene, Light{Vec3(10, 10, -10), Vec3(1, 1, 1), 0.7});
    add_light(scene, Light{Vec3(-10, 10, -10), Vec3(1, 1, 0.8), 0.5});

    // Set ambient
    set_ambient(scene, Vec3(0.1, 0.1, 0.1));

    return scene;
}

bool load_scene_hybrid(const std::string& filename, Scene& scene,
                      Camera& camera, int width, int height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open scene file: " << filename << std::endl;
        return false;
    }

    std::string line;
    Vec3 cam_pos(0, 2, 5);
    Vec3 cam_lookat(0, 0, -20);
    double cam_fov = 60.0;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "sphere") {
            double x, y, z, radius, r, g, b, metallic, roughness, shininess;
            if (iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >> roughness >> shininess) {
                add_sphere(scene, Sphere(Vec3(x, y, z), radius,
                                       Material{Vec3(r, g, b), metallic, shininess}));
            }
        } else if (type == "light") {
            double x, y, z, r, g, b, intensity;
            if (iss >> x >> y >> z >> r >> g >> b >> intensity) {
                add_light(scene, Light{Vec3(x, y, z), Vec3(r, g, b), intensity});
            }
        } else if (type == "ambient") {
            double r, g, b;
            if (iss >> r >> g >> b) {
                set_ambient(scene, Vec3(r, g, b));
            }
        } else if (type == "camera") {
            double px, py, pz, lx, ly, lz, fov;
            if (iss >> px >> py >> pz >> lx >> ly >> lz >> fov) {
                cam_pos = Vec3(px, py, pz);
                cam_lookat = Vec3(lx, ly, lz);
                cam_fov = fov;
            }
        }
    }

    file.close();

    // Create camera
    camera = Camera(cam_pos, cam_lookat, cam_fov);

    std::cout << "Loaded scene from " << filename << ":\n";
    std::cout << "  Spheres: " << scene.spheres.size() << "\n";
    std::cout << "  Lights: " << scene.lights.size() << "\n";

    return true;
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
// Scene Helper Methods
// =========================================================
namespace {
    const std::vector<Sphere>& get_spheres(const Scene& scene) {
        return scene.spheres;
    }

    const std::vector<Light>& get_lights(const Scene& scene) {
        return scene.lights;
    }

    const Vec3& get_ambient(const Scene& scene) {
        return scene.ambient_light;
    }

    void add_sphere(Scene& scene, const Sphere& sphere) {
        scene.spheres.push_back(sphere);
    }

    void add_light(Scene& scene, const Light& light) {
        scene.lights.push_back(light);
    }

    void set_ambient(Scene& scene, const Vec3& ambient) {
        scene.ambient_light = ambient;
    }

    void print_stats(const Scene& scene) {
        std::cout << "Scene stats:\n";
        std::cout << "  Spheres: " << scene.spheres.size() << "\n";
        std::cout << "  Lights: " << scene.lights.size() << "\n";
    }

    Vec3 get_background(const Scene& scene, const Ray& ray) {
        double t = 0.5 * (ray.direction.y + 1.0);
        return Vec3(1, 1, 1) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
    }
}

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

extern "C" void upload_lights_to_constant(float *h_lights, int num_lights, float *h_ambient);

// =========================================================
// CPU Ray Tracing (Complex Shading Path)
// =========================================================
Vec3 trace_ray_cpu(const Ray &ray, const Scene &scene, int depth)
{
    if (depth <= 0)
        return Vec3(0, 0, 0);

    double t;
    int sphere_idx;

    // Find closest intersection
    if (!scene.find_intersection(ray, t, sphere_idx))
    {
        // Sky color gradient background
        double blend = 0.5 * (ray.direction.y + 1.0);
        return Vec3(1, 1, 1) * (1.0 - blend) + Vec3(0.5, 0.7, 1.0) * blend;
    }

    // Calculate hit point and normal
    const Sphere &sphere = scene.spheres[sphere_idx];
    Vec3 hit_point = ray.at(t);
    Vec3 normal = sphere.normal_at(hit_point);
    Vec3 view_dir = (ray.origin - hit_point).normalized();

    // Get shaded color
    Vec3 color = scene.shade(hit_point, normal, sphere.material, view_dir);

    // Handle reflections
    if (sphere.material.reflectivity > 0.01 && depth > 1)
    {
        Vec3 reflect_dir = reflect(ray.direction, normal);
        Ray reflect_ray(hit_point + normal * 0.001, reflect_dir);
        Vec3 reflect_color = trace_ray_cpu(reflect_ray, scene, depth - 1);

        // Mix direct color with reflection
        color = color * (1.0 - sphere.material.reflectivity) +
                reflect_color * sphere.material.reflectivity;
    }

    return color;
}

void process_tile_cpu(const Tile &tile, const Scene &scene, const Camera &camera,
                      std::vector<Vec3> &framebuffer, int width, int height, int max_depth)
{

#pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int y = tile.y_start; y < tile.y_end; y++)
    {
        for (int x = tile.x_start; x < tile.x_end; x++)
        {
            double u = double(x) / double(width - 1);
            double v = double(y) / double(height - 1);
            Ray ray = camera.get_ray(u, v);
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
        spheres_size = num_spheres * 10 * sizeof(float); // center(3) + radius(1) + albedo(3) + metallic(1) + roughness(1) + shininess(1)
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
        // Pack spheres into flat array: [center.xyz, radius, albedo.xyz, metallic, roughness, shininess]
        const auto& spheres = get_spheres(scene);
        std::vector<float> h_spheres_packed(spheres.size() * 10);
        for (size_t i = 0; i < spheres.size(); i++) {
            int idx = i * 10;
            h_spheres_packed[idx + 0] = spheres[i].center.x;
            h_spheres_packed[idx + 1] = spheres[i].center.y;
            h_spheres_packed[idx + 2] = spheres[i].center.z;
            h_spheres_packed[idx + 3] = spheres[i].radius;
            h_spheres_packed[idx + 4] = spheres[i].material.color.x;
            h_spheres_packed[idx + 5] = spheres[i].material.color.y;
            h_spheres_packed[idx + 6] = spheres[i].material.color.z;
            h_spheres_packed[idx + 7] = spheres[i].material.reflectivity;
            h_spheres_packed[idx + 8] = 0.0f; // roughness (not used in current material)
            h_spheres_packed[idx + 9] = spheres[i].material.shininess;
        }

        // Pack lights into flat array: [position.xyz, color.xyz, intensity]
        const auto& lights = get_lights(scene);
        std::vector<float> h_lights_packed(lights.size() * 7);
        for (size_t i = 0; i < lights.size(); i++) {
            int idx = i * 7;
            h_lights_packed[idx + 0] = lights[i].position.x;
            h_lights_packed[idx + 1] = lights[i].position.y;
            h_lights_packed[idx + 2] = lights[i].position.z;
            h_lights_packed[idx + 3] = lights[i].color.x;
            h_lights_packed[idx + 4] = lights[i].color.y;
            h_lights_packed[idx + 5] = lights[i].color.z;
            h_lights_packed[idx + 6] = lights[i].intensity;
        }

        // Pack ambient light
        const Vec3& ambient = get_ambient(scene);
        float h_ambient[3] = {(float)ambient.x, (float)ambient.y, (float)ambient.z};

        // Upload to GPU
        if (spheres.size() > 0) {
            cudaMemcpy(d_spheres, h_spheres_packed.data(), spheres_size, cudaMemcpyHostToDevice);
        }
        if (lights.size() > 0) {
            cudaMemcpy(d_lights, h_lights_packed.data(), lights_size, cudaMemcpyHostToDevice);
        }

        // Upload lights to constant memory
        upload_lights_to_constant(h_lights_packed.data(), lights.size(), h_ambient);
    }

    void download_tile(const Tile &tile, std::vector<Vec3> &framebuffer,
                       int width, cudaStream_t stream)
    {
        // Download tile region from GPU framebuffer to CPU framebuffer
        // Copy row by row since the tile may not span full width
        for (int y = tile.y_start; y < tile.y_end; y++)
        {
            int row_offset = y * width + tile.x_start;
            int row_width = tile.x_end - tile.x_start;

            // Create temporary buffer for row
            std::vector<float> row_data(row_width * 3);

            // Download row from GPU (async)
            cudaMemcpyAsync(row_data.data(),
                           d_framebuffer + row_offset * 3,
                           row_width * 3 * sizeof(float),
                           cudaMemcpyDeviceToHost,
                           stream);

            // Synchronize stream to ensure data is ready
            cudaStreamSynchronize(stream);

            // Unpack into Vec3 framebuffer
            for (int x = 0; x < row_width; x++)
            {
                int fb_idx = y * width + tile.x_start + x;
                framebuffer[fb_idx] = Vec3(row_data[x * 3 + 0],
                                          row_data[x * 3 + 1],
                                          row_data[x * 3 + 2]);
            }
        }
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
            // Pack camera parameters into flat array
            float camera_params[12];
            camera_params[0] = camera.position.x;
            camera_params[1] = camera.position.y;
            camera_params[2] = camera.position.z;

            // Calculate camera basis vectors (similar to GPU version)
            Vec3 forward = (Vec3(0, 0, -20) - camera.position).normalized();
            Vec3 right = cross(forward, Vec3(0, 1, 0)).normalized();
            Vec3 up = cross(right, forward).normalized();

            double aspect = double(width) / double(height);
            double scale = tan(camera.fov * 0.5 * M_PI / 180.0);

            Vec3 horizontal = right * (2.0 * scale * aspect);
            Vec3 vertical = up * (2.0 * scale);
            Vec3 lower_left = camera.position + forward - horizontal * 0.5 - vertical * 0.5;

            camera_params[3] = lower_left.x;
            camera_params[4] = lower_left.y;
            camera_params[5] = lower_left.z;
            camera_params[6] = horizontal.x;
            camera_params[7] = horizontal.y;
            camera_params[8] = horizontal.z;
            camera_params[9] = vertical.x;
            camera_params[10] = vertical.y;
            camera_params[11] = vertical.z;

            int stream_idx = 0;
            std::vector<Tile*> processing_tiles;

            while (!gpu_queue.empty())
            {
                Tile *tile = gpu_queue.front();
                gpu_queue.pop();

                cudaStream_t stream = streams[stream_idx];
                stream_idx = (stream_idx + 1) % NUM_STREAMS;

                // Launch GPU kernel for this tile
                launch_gpu_kernel(
                    gpu_resources.get_framebuffer(),
                    gpu_resources.get_spheres(),
                    scene.spheres.size(),
                    gpu_resources.get_lights(),
                    scene.lights.size(),
                    camera_params,
                    tile->x_start, tile->y_start,
                    tile->x_end - tile->x_start,
                    tile->y_end - tile->y_start,
                    width, height,
                    max_depth,
                    stream
                );

                // Download results asynchronously
                gpu_resources.download_tile(*tile, framebuffer, width, stream);

                processing_tiles.push_back(tile);
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
    Scene scene = create_test_scene();
    print_stats(scene);

    // Setup camera
    Vec3 lookfrom(0, 2, 5);
    Vec3 lookat(0, 0, -20);
    double vfov = 60.0;

    Camera camera(lookfrom, lookat, vfov);

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
