#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
// #include <omp.h>
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

int main(int argc, char *argv[])
{
    // Image settings
    const int width = 640;
    const int height = 480;
    const int max_depth = 3;

    // Create scene
    Scene scene;

    // Add spheres with different materials
    // Center: Red diffuse sphere
    scene.spheres.push_back(Sphere(Vec3(0, 0, -20), 3,
                                   Material{Vec3(1, 0, 0), 0.2, 50}));

    // Left: Green sphere (more diffuse)
    scene.spheres.push_back(Sphere(Vec3(-6, -1, -18), 2,
                                   Material{Vec3(0, 1, 0), 0.1, 30}));

    // Right: Blue shiny sphere
    scene.spheres.push_back(Sphere(Vec3(5, 0, -17), 2,
                                   Material{Vec3(0.3, 0.3, 1), 0.6, 100}));

    // Small mirror sphere in front
    scene.spheres.push_back(Sphere(Vec3(2, -2, -12), 1.5,
                                   Material{Vec3(0.9, 0.9, 0.9), 0.8, 200}));

    // Ground plane (large sphere below)
    scene.spheres.push_back(Sphere(Vec3(0, -1004, -20), 1000,
                                   Material{Vec3(0.5, 0.5, 0.5), 0.1, 10}));

    // Add lights
    // Main light (top right)
    scene.lights.push_back(Light{Vec3(10, 10, -10), Vec3(1, 1, 1), 0.8});

    // Fill light (left side, softer)
    scene.lights.push_back(Light{Vec3(-10, 5, -5), Vec3(0.5, 0.5, 0.7), 0.3});

    // Setup camera
    Camera camera(Vec3(0, 0, 0), Vec3(0, 0, -1), 60);

    // Framebuffer
    std::vector<Vec3> framebuffer(width * height);

    // Timing
    auto start = std::chrono::high_resolution_clock::now();

    // SERIAL VERSION
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Serial time: " << diff.count() << " seconds\n";

    write_ppm("output_serial.ppm", framebuffer, width, height);

// TODO: STUDENT - Add OpenMP version
// OPENMP VERSION
#ifdef _OPENMP
    std::cout << "\nRendering (OpenMP)...\n";
    start = std::chrono::high_resolution_clock::now();

    // YOUR OPENMP CODE HERE
    // Hint: Use #pragma omp parallel for with appropriate scheduling

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "OpenMP time: " << diff.count() << " seconds\n";

    write_ppm("output_openmp.ppm", framebuffer, width, height);
#endif

    return 0;
}