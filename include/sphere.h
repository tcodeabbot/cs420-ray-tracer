#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"

struct Material
{
    Vec3 color;          // Base color
    double reflectivity; // 0.0 = diffuse, 1.0 = mirror
    double shininess;    // Phong exponent
};

class Sphere
{
public:
    Vec3 center;
    double radius;
    Material material;

    Sphere(Vec3 c, double r, Material m) : center(c), radius(r), material(m) {}

    // TODO: STUDENT IMPLEMENTATION
    // Implement ray-sphere intersection test
    // Return true if ray hits sphere, store distance in t
    // Hint: Solve quadratic equation from ray equation and sphere equation
    bool intersect(const Ray &ray, double &t) const
    {
        // YOUR CODE HERE
        // 1. Calculate discriminant
        // 2. Check if discriminant >= 0
        // 3. Calculate t values
        // 4. Return smallest positive t

        // Vector from ray origin to sphere center
        Vec3 oc = ray.origin - center;

        // Coefficients for the quadratic equation
        double a = ray.direction.dot(ray.direction);
        double b = 2.0 * oc.dot(ray.direction);
        double c = oc.dot(oc) - radius * radius;

        // Discriminant
        double discriminant = b * b - 4 * a * c;

        if (discriminant < 0)
        {
            // No intersection
            return false;
        }
        else
        {
            // Calculate the two possible t values
            double t1 = (-b - sqrt(discriminant)) / (2 * a);
            double t2 = (-b + sqrt(discriminant)) / (2 * a);

            // Find the smallest positive t
            if (t1 > 0 && t2 > 0)
            {
                t = std::min(t1, t2);
            }
            else if (t1 > 0)
            {
                t = t1;
            }
            else if (t2 > 0)
            {
                t = t2;
            }
            else
            {
                // Both t1 and t2 are negative, no intersection
                return false;
            }

            return true;
        }
    }

    // Calculate normal at point on sphere surface
    Vec3 normal_at(const Vec3 &point) const
    {
        return (point - center).normalized();
    }
};

#endif