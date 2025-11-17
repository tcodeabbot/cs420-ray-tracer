# CS420 Ray Tracer - UNIX/Linux Package

## ✅ For UNIX/Linux Systems - Use These Files

Since you're running on UNIX machines, here's exactly what you need:

### Core Files to Use

```bash
# Main build system
Makefile_unix → rename to Makefile

# Headers (already clean, no Windows-specific code needed)
include/vec3.h        # Use as-is (or vec3_unix.h if issues)
include/ray.h         # Use as-is
include/camera.h      # Use as-is (or camera_unix.h if issues) 
include/sphere.h      # Student template
include/material.h    # Use material_unix.h (no hittable.h dependency)
include/scene.h       # Student template

# Source files
src/main.cpp          # Use main_unix.cpp (clean version)
src/main_gpu.cu       # Week 2 CUDA template
src/main_hybrid.cpp   # Week 3 hybrid template

# Scenes
scenes/simple.txt     # 5 spheres test scene
scenes/medium.txt     # 50 spheres test scene
scenes/complex.txt    # 200 spheres test scene

# Scripts (already UNIX bash scripts)
scripts/test.sh       # Automated testing
scripts/benchmark.sh  # Performance measurement
```

### Quick Setup Commands

```bash
# 1. Use the UNIX Makefile
mv Makefile_unix Makefile

# 2. Use clean UNIX versions of files
cp include/material_unix.h include/material.h
cp include/vec3_unix.h include/vec3.h
cp include/camera_unix.h include/camera.h
cp src/main_unix.cpp src/main.cpp

# 3. Build everything
make clean all

# 4. Test
make test
make benchmark
```

### What's Different in UNIX Version

1. **No Windows-specific defines** - M_PI works directly
2. **Clean headers** - No `_USE_MATH_DEFINES` needed
3. **Proper OpenMP handling** - Wrapped with `#ifdef _OPENMP`
4. **No hittable.h dependency** - material.h is self-contained
5. **Comprehensive Makefile** - All targets, profiling, and utilities

### Files You DON'T Need (Windows-specific)

- ❌ math_constants.h (not needed on UNIX)
- ❌ CS420_RayTracer.vcxproj 
- ❌ build.bat
- ❌ test_setup.cpp (Windows test)
- ❌ VS_BUILD.md
- ❌ INTELLISENSE_FIX.md
- ❌ .vscode/ folder

### Complete UNIX Makefile Features

The `Makefile_unix` provides everything you need:

```makefile
# Building
make serial       # Week 1: Serial version
make openmp       # Week 1: OpenMP version  
make cuda         # Week 2: GPU version
make hybrid       # Week 3: Hybrid version
make all          # Build everything

# Testing
make test         # Quick test
make test-all     # Test all versions
make benchmark    # Performance comparison

# Profiling
make profile-serial   # gprof profiling
make profile-openmp   # perf profiling
make profile-cuda     # nvprof profiling

# Memory checking
make memcheck         # valgrind leak check
make memcheck-openmp  # helgrind race detection
make memcheck-cuda    # cuda-memcheck

# Utilities
make clean        # Remove all artifacts
make view         # Display output image
make convert      # PPM to PNG conversion
make check-env    # Verify environment
make help         # Show all commands
```

### Instructor Distribution Package

For student distribution, provide these files:

```
cs420-ray-tracer.tar.gz containing:
│
├── Makefile              # (renamed from Makefile_unix)
├── README.md             # Student instructions
├── include/
│   ├── vec3.h
│   ├── ray.h
│   ├── camera.h
│   ├── sphere.h         # Template with TODOs
│   ├── material.h       # Template with TODOs
│   └── scene.h          # Template with TODOs
├── src/
│   ├── main.cpp         # Template with TODOs
│   ├── main_gpu.cu      # Week 2 template
│   └── main_hybrid.cpp  # Week 3 template
├── scenes/
│   ├── simple.txt
│   ├── medium.txt
│   └── complex.txt
└── scripts/
    ├── test.sh
    └── benchmark.sh
```

### Create Distribution Archive

```bash
# Package for students
tar -czf cs420_ray_tracer.tar.gz \
    Makefile README.md \
    include/ src/ scenes/ scripts/

# Students extract with
tar -xzf cs420_ray_tracer.tar.gz
cd cs420-ray-tracer
make check-env
make all
```

### Testing on Different UNIX Systems

The package works on:
- ✅ Ubuntu/Debian Linux
- ✅ CentOS/RHEL/Fedora  
- ✅ macOS (with minor adjustments)
- ✅ WSL/WSL2 on Windows
- ✅ Any system with g++ 5.0+ and make

### Dependencies

Minimal requirements:
- g++ or clang++ (C++11 support)
- make
- OpenMP (libomp-dev on Ubuntu)

Optional:
- CUDA toolkit (for GPU version)
- valgrind (for memory checking)
- perf (for profiling)
- ImageMagick (for PPM conversion)

### Summary

You have a complete, clean UNIX package that:
1. **Builds properly** with a comprehensive Makefile
2. **No Windows cruft** - pure UNIX/Linux focus
3. **All weeks covered** - Serial, OpenMP, CUDA, Hybrid
4. **Testing included** - Automated scripts
5. **Student-friendly** - Clear TODOs and structure

The `Makefile_unix` is production-ready and handles all edge cases including CUDA detection, proper dependency tracking, and multiple build configurations.

Use `make help` for complete command reference.
