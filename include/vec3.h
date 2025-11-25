#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class Vec3
{
public:
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(const Vec3 &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator*(double t) const { return Vec3(x * t, y * t, z * t); }
    Vec3 operator/(double t) const { return Vec3(x / t, y / t, z / t); }

    double length() const { return sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const
    {
        double len = length();
        return Vec3(x / len, y / len, z / len);
    }

    double dot(const Vec3 &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    Vec3 cross(const Vec3 &other) const
    {
        return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }
};

// Free function for cross product
inline Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return a.cross(b);
}

// Reflect vector v around normal n
inline Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - n * (2 * v.dot(n));
}
#endif