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