#!/bin/bash
# test.sh - Automated testing script for CS420 Ray Tracer
# Usage: ./test.sh [serial|openmp|cuda|all]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
WIDTH=320
HEIGHT=240
TIMEOUT=30

echo "========================================="
echo "CS420 Ray Tracer - Automated Test Suite"
echo "========================================="
echo ""

# Function to run a test
run_test() {
    local name=$1
    local executable=$2
    local args=$3
    local expected_output=$4
    
    echo -n "Testing $name... "
    
    if [ ! -f "$executable" ]; then
        echo -e "${RED}SKIP${NC} (not built)"
        return 1
    fi
    
    # Run with timeout
    timeout $TIMEOUT ./$executable $args > test_output.log 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo -e "${RED}TIMEOUT${NC} (>${TIMEOUT}s)"
        return 1
    elif [ $exit_code -ne 0 ]; then
        echo -e "${RED}FAILED${NC} (exit code: $exit_code)"
        cat test_output.log
        return 1
    fi
    
    # Check if output file was created
    if [ ! -f "$expected_output" ]; then
        echo -e "${RED}FAILED${NC} (no output file)"
        return 1
    fi
    
    # Check file size (basic validation)
    local filesize=$(stat -c%s "$expected_output" 2>/dev/null || stat -f%z "$expected_output" 2>/dev/null)
    if [ "$filesize" -lt 1000 ]; then
        echo -e "${RED}FAILED${NC} (output too small)"
        return 1
    fi
    
    # Extract timing if available
    local timing=$(grep -E "(time:|seconds)" test_output.log | head -1)
    if [ -n "$timing" ]; then
        echo -e "${GREEN}PASSED${NC} ($timing)"
    else
        echo -e "${GREEN}PASSED${NC}"
    fi
    
    return 0
}

# Function to compare images (basic check)
compare_images() {
    local img1=$1
    local img2=$2
    
    if [ ! -f "$img1" ] || [ ! -f "$img2" ]; then
        return 1
    fi
    
    # Compare file sizes
    local size1=$(stat -c%s "$img1" 2>/dev/null || stat -f%z "$img1" 2>/dev/null)
    local size2=$(stat -c%s "$img2" 2>/dev/null || stat -f%z "$img2" 2>/dev/null)
    
    local diff=$((size1 - size2))
    if [ $diff -lt 0 ]; then
        diff=$((-diff))
    fi
    
    # Allow 5% difference in file size
    local threshold=$((size1 / 20))
    if [ $diff -gt $threshold ]; then
        echo "  Warning: Output files differ significantly in size"
        return 1
    fi
    
    return 0
}

# Performance test
performance_test() {
    echo ""
    echo "Performance Comparison:"
    echo "-----------------------"
    
    local serial_time=""
    local openmp_time=""
    local cuda_time=""
    
    if [ -f "ray_serial" ]; then
        echo -n "Serial: "
        serial_time=$( { time -p ./ray_serial > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
        echo "${serial_time}s"
    fi
    
    if [ -f "ray_openmp" ]; then
        echo -n "OpenMP (4 threads): "
        openmp_time=$( { time -p OMP_NUM_THREADS=4 ./ray_openmp --openmp > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
        echo "${openmp_time}s"
        
        if [ -n "$serial_time" ] && [ -n "$openmp_time" ]; then
            speedup=$(echo "scale=2; $serial_time / $openmp_time" | bc)
            echo "  Speedup: ${speedup}x"
            
            # Check if speedup meets requirement
            if (( $(echo "$speedup >= 2.5" | bc -l) )); then
                echo -e "  ${GREEN}✓ Meets 2.5x requirement${NC}"
            else
                echo -e "  ${YELLOW}⚠ Below 2.5x requirement${NC}"
            fi
        fi
    fi
    
    if [ -f "ray_cuda" ]; then
        echo -n "CUDA: "
        cuda_time=$( { time -p ./ray_cuda > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
        echo "${cuda_time}s"
        
        if [ -n "$serial_time" ] && [ -n "$cuda_time" ]; then
            speedup=$(echo "scale=2; $serial_time / $cuda_time" | bc)
            echo "  Speedup: ${speedup}x"
            
            # Check if speedup meets requirement
            if (( $(echo "$speedup >= 10" | bc -l) )); then
                echo -e "  ${GREEN}✓ Meets 10x requirement${NC}"
            else
                echo -e "  ${YELLOW}⚠ Below 10x requirement${NC}"
            fi
        fi
    fi
}

# Memory check
memory_check() {
    echo ""
    echo "Memory Check:"
    echo "-------------"
    
    if command -v valgrind > /dev/null; then
        if [ -f "ray_serial" ]; then
            echo "Checking serial version..."
            valgrind --leak-check=summary --error-exitcode=1 ./ray_serial > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "  ${GREEN}✓ No memory leaks detected${NC}"
            else
                echo -e "  ${RED}✗ Memory issues detected${NC}"
                echo "  Run 'make memcheck-serial' for details"
            fi
        fi
    else
        echo "  Valgrind not installed (skipping memory check)"
    fi
}

# Main test execution
TEST_TYPE=${1:-all}

echo "Test Configuration:"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Timeout: ${TIMEOUT}s"
echo "  Test type: $TEST_TYPE"
echo ""

case $TEST_TYPE in
    serial)
        run_test "Serial Implementation" "ray_serial" "" "output_serial.ppm"
        ;;
    openmp)
        run_test "OpenMP Implementation" "ray_openmp" "--openmp" "output_openmp.ppm"
        ;;
    cuda)
        run_test "CUDA Implementation" "ray_cuda" "" "output_gpu.ppm"
        ;;
    all)
        echo "Running All Tests:"
        echo "------------------"
        run_test "Serial Implementation" "ray_serial" "" "output_serial.ppm"
        run_test "OpenMP Implementation" "ray_openmp" "--openmp" "output_openmp.ppm"
        run_test "CUDA Implementation" "ray_cuda" "" "output_gpu.ppm"
        
        echo ""
        echo "Image Comparison:"
        echo "-----------------"
        if [ -f "output_serial.ppm" ] && [ -f "output_openmp.ppm" ]; then
            compare_images "output_serial.ppm" "output_openmp.ppm"
            if [ $? -eq 0 ]; then
                echo -e "  ${GREEN}✓ Serial and OpenMP outputs match${NC}"
            else
                echo -e "  ${YELLOW}⚠ Serial and OpenMP outputs differ${NC}"
            fi
        fi
        
        performance_test
        memory_check
        ;;
    *)
        echo "Usage: $0 [serial|openmp|cuda|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Test Summary:"

# Count results
PASSED=$(grep -c "PASSED" test_output.log 2>/dev/null || echo 0)
FAILED=$(grep -c "FAILED" test_output.log 2>/dev/null || echo 0)

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${YELLOW}Some tests failed. Please review output above.${NC}"
fi

# Cleanup
rm -f test_output.log

echo "========================================="
