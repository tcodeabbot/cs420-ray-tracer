#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <algorithm>
#include <cmath>
#include "sphere.h"
#include "vec3.h"
#include "ray.h"

struct Light {
    Vec3 position;
    Vec3 color;
    double intensity;
};

class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Light> lights;
    Vec3 ambient_light;
    
    Scene() : ambient_light(0.1, 0.1, 0.1) {}
    
    // Find closest sphere intersection
    bool find_intersection(const Ray& ray, double& t, int& sphere_idx) const {
        t = INFINITY;
        sphere_idx = -1;

        // Loop through all spheres and find the closest intersection
        for (size_t i = 0; i < spheres.size(); i++) {
            double t_temp;
            if (spheres[i].intersect(ray, t_temp)) {
                if (t_temp < t) {
                    t = t_temp;
                    sphere_idx = i;
                }
            }
        }

        return sphere_idx >= 0;
    }
    
    // Check if point is in shadow from light
    bool in_shadow(const Vec3& point, const Light& light) const {
        // Cast ray from point to light
        Vec3 to_light = light.position - point;
        double light_distance = to_light.length();
        Vec3 shadow_dir = to_light.normalized();

        Ray shadow_ray(point, shadow_dir);

        // Check for intersection with any sphere
        for (size_t i = 0; i < spheres.size(); i++) {
            double t;
            if (spheres[i].intersect(shadow_ray, t)) {
                // If intersection is between point and light, we're in shadow
                if (t > 0.001 && t < light_distance) {
                    return true;
                }
            }
        }

        return false;
    }
    
    // Calculate color at intersection point using Phong shading
    Vec3 shade(const Vec3& point, const Vec3& normal, const Material& mat,
               const Vec3& view_dir) const {
        Vec3 color = ambient_light * mat.color;

        // For each light in the scene
        for (const Light& light : lights) {
            // Check if point is in shadow
            if (in_shadow(point, light)) {
                continue;  // Skip this light if in shadow
            }

            // Light direction from point to light
            Vec3 light_dir = (light.position - point).normalized();

            // Diffuse component (Lambert)
            double n_dot_l = std::max(0.0, normal.dot(light_dir));
            Vec3 diffuse = mat.color * light.color * light.intensity * n_dot_l;

            // Specular component (Phong)
            Vec3 reflect_dir = reflect(light_dir * -1.0, normal);
            double r_dot_v = std::max(0.0, reflect_dir.dot(view_dir));
            double spec_factor = std::pow(r_dot_v, mat.shininess);
            Vec3 specular = light.color * light.intensity * mat.reflectivity * spec_factor;

            color = color + diffuse + specular;
        }

        return color;
    }
};

#endif