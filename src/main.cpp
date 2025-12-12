#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <sstream>
#include <string>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "scene.h"
#include <math_constants.h>

class Camera
{
public:
    Vec3 position;
    Vec3 forward, right, up;
    double fov;

    Camera(Vec3 pos, Vec3 look_at, double field_of_view)
        : position(pos), fov(field_of_view)
    {
        forward = (look_at - position).normalized();
        right = cross(forward, Vec3(0, 1, 0)).normalized();
        up = cross(right, forward).normalized();
    }

    Ray get_ray(double u, double v) const
    {
        double aspect = 1.0;
        double scale = tan(fov * 0.5 * M_PI / 180.0);

        Vec3 direction = forward + right * ((u - 0.5) * scale * aspect) + up * ((v - 0.5) * scale);

        return Ray(position, direction.normalized());
    }
};

// Trace a single ray through the scene
Vec3 trace_ray(const Ray &ray, const Scene &scene, int depth)
{
    if (depth <= 0)
        return Vec3(0, 0, 0);

    double t;
    int sphere_idx;

    if (!scene.find_intersection(ray, t, sphere_idx))
    {
        // Sky color gradient
        double t = 0.5 * (ray.direction.y + 1.0);
        return Vec3(1, 1, 1) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t;
    }

    // Calculate hit point and normal
    const Sphere &sphere = scene.spheres[sphere_idx];
    Vec3 hit_point = ray.origin + ray.direction * t;
    Vec3 normal = sphere.normal_at(hit_point);
    Vec3 view_dir = (ray.origin - hit_point).normalized();

    // Get color from shading
    Vec3 color = scene.shade(hit_point, normal, sphere.material, view_dir);

    // Handle reflections
    if (sphere.material.reflectivity > 0.01 && depth > 1)
    {
        Vec3 reflect_dir = reflect(ray.direction, normal);
        Ray reflect_ray(hit_point + normal * 0.001, reflect_dir); // Offset to avoid self-intersection
        Vec3 reflect_color = trace_ray(reflect_ray, scene, depth - 1);

        // Mix shaded color with reflection
        color = color * (1.0 - sphere.material.reflectivity) + reflect_color * sphere.material.reflectivity;
    }

    return color;
}

// Write image to PPM file
void write_ppm(const std::string &filename, const std::vector<Vec3> &framebuffer,
               int width, int height)
{
    std::ofstream file(filename);
    file << "P3\n"
         << width << " " << height << "\n255\n";

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
}

// Load scene from file
// Returns true on success, false on failure
// Also returns camera parameters via reference
bool load_scene(const std::string &filename, Scene &scene,
                Vec3 &cam_pos, Vec3 &cam_lookat, double &cam_fov)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open scene file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int line_num = 0;

    // Set defaults
    cam_pos = Vec3(0, 0, 0);
    cam_lookat = Vec3(0, 0, -1);
    cam_fov = 60;
    scene.ambient_light = Vec3(0.1, 0.1, 0.1);

    while (std::getline(file, line))
    {
        line_num++;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "sphere")
        {
            // Format: sphere x y z radius r g b metallic roughness shininess
            double x, y, z, radius, r, g, b, metallic, roughness, shininess;
            if (iss >> x >> y >> z >> radius >> r >> g >> b >> metallic >> roughness >> shininess)
            {
                // Note: The scene file format uses metallic/roughness, but our Material uses reflectivity
                // We'll map metallic to reflectivity for now
                scene.spheres.push_back(Sphere(
                    Vec3(x, y, z),
                    radius,
                    Material{Vec3(r, g, b), metallic, shininess}
                ));
            }
            else
            {
                std::cerr << "Warning: Invalid sphere format at line " << line_num << std::endl;
            }
        }
        else if (type == "light")
        {
            // Format: light x y z r g b intensity
            double x, y, z, r, g, b, intensity;
            if (iss >> x >> y >> z >> r >> g >> b >> intensity)
            {
                scene.lights.push_back(Light{Vec3(x, y, z), Vec3(r, g, b), intensity});
            }
            else
            {
                std::cerr << "Warning: Invalid light format at line " << line_num << std::endl;
            }
        }
        else if (type == "ambient")
        {
            // Format: ambient r g b
            double r, g, b;
            if (iss >> r >> g >> b)
            {
                scene.ambient_light = Vec3(r, g, b);
            }
            else
            {
                std::cerr << "Warning: Invalid ambient format at line " << line_num << std::endl;
            }
        }
        else if (type == "camera")
        {
            // Format: camera pos_x pos_y pos_z lookat_x lookat_y lookat_z fov
            double px, py, pz, lx, ly, lz, fov;
            if (iss >> px >> py >> pz >> lx >> ly >> lz >> fov)
            {
                cam_pos = Vec3(px, py, pz);
                cam_lookat = Vec3(lx, ly, lz);
                cam_fov = fov;
            }
            else
            {
                std::cerr << "Warning: Invalid camera format at line " << line_num << std::endl;
            }
        }
        else
        {
            std::cerr << "Warning: Unknown type '" << type << "' at line " << line_num << std::endl;
        }
    }

    file.close();

    std::cout << "Loaded scene from " << filename << ":\n";
    std::cout << "  Spheres: " << scene.spheres.size() << "\n";
    std::cout << "  Lights: " << scene.lights.size() << "\n";

    return true;
}

int main(int argc, char *argv[])
{
    // Parse command-line arguments
    std::string scene_file = "scenes/simple.txt"; // Default scene

    if (argc > 1)
    {
        scene_file = argv[1];
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [scene_file]\n";
        std::cout << "Using default scene: " << scene_file << "\n\n";
    }

    // Image settings
    const int width = 640;
    const int height = 480;
    const int max_depth = 3;

    // Create scene and load from file
    Scene scene;
    Vec3 cam_pos, cam_lookat;
    double cam_fov;

    if (!load_scene(scene_file, scene, cam_pos, cam_lookat, cam_fov))
    {
        std::cerr << "Failed to load scene. Exiting.\n";
        return 1;
    }

    // Setup camera from loaded parameters
    Camera camera(cam_pos, cam_lookat, cam_fov);

    // Framebuffer
    std::vector<Vec3> framebuffer(width * height);

    // Timing variables
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    std::chrono::duration<double> diff;

#ifndef _OPENMP
    // SERIAL VERSION (only when OpenMP is not available)
    std::cout << "Rendering (Serial)...\n";
    for (int j = 0; j < height; j++)
    {
        if (j % 50 == 0)
            std::cout << "Row " << j << "/" << height << "\n";

        for (int i = 0; i < width; i++)
        {
            double u = double(i) / (width - 1);
            double v = double(j) / (height - 1);

            Ray ray = camera.get_ray(u, v);
            framebuffer[j * width + i] = trace_ray(ray, scene, max_depth);
        }
    }

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Serial time: " << diff.count() << " seconds\n";

    write_ppm("output_serial.ppm", framebuffer, width, height);
#else
    // OPENMP VERSION
    std::cout << "Rendering (OpenMP)...\n";
    std::cout << "Using " << omp_get_max_threads() << " threads\n";

// Optimized parallel ray tracing with OpenMP
// - schedule(dynamic, 1): Fine-grained dynamic scheduling for optimal load balancing
//   * Chunk size of 1 row ensures no thread sits idle while others are still working
//   * Critical for workloads with varying pixel complexity (reflections, shadows)
//   * Small overhead per chunk is acceptable for compute-intensive ray tracing
// - Parallelize outer loop only for better cache locality
// - Each pixel writes to unique framebuffer location (no race conditions)
#pragma omp parallel for num_threads(8) schedule(dynamic, 1)
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            double u = double(i) / (width - 1);
            double v = double(j) / (height - 1);

            Ray ray = camera.get_ray(u, v);
            framebuffer[j * width + i] = trace_ray(ray, scene, max_depth);
        }
    }

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "OpenMP time: " << diff.count() << " seconds\n";

    write_ppm("output_openmp.ppm", framebuffer, width, height);
#endif

    return 0;
}