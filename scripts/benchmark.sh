#!/bin/bash
# benchmark.sh - Performance benchmarking script for CS420 Ray Tracer
# Measures and compares performance across all implementations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCENES=("simple.txt" "medium.txt" "complex.txt")
THREAD_COUNTS=(1 2 4 8)
ITERATIONS=3
OUTPUT_DIR="benchmark_results"

echo "========================================="
echo "CS420 Ray Tracer - Performance Benchmark"
echo "========================================="
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$OUTPUT_DIR/benchmark_${TIMESTAMP}.csv"

# Write CSV header
echo "Implementation,Scene,Threads,Iteration,Time(s),Pixels/s,Speedup" > $RESULTS_FILE

# Function to extract time from program output
extract_time() {
    grep -E "(time:|seconds)" | head -1 | grep -oE "[0-9]+\.[0-9]+"
}

# Function to calculate pixels per second
calc_pixels_per_second() {
    local time=$1
    local width=$2
    local height=$3
    echo "scale=2; ($width * $height) / $time" | bc
}

# Function to run a single benchmark
run_benchmark() {
    local executable=$1
    local scene=$2
    local threads=$3
    local iteration=$4
    local name=$5
    
    if [ ! -f "$executable" ]; then
        echo -e "  ${YELLOW}Skipping${NC} $name (not built)"
        return
    fi
    
    echo -n "  $name (threads=$threads, iter=$iteration)... "
    
    # Set thread count for OpenMP
    export OMP_NUM_THREADS=$threads
    
    # Run and measure time
    start_time=$(date +%s.%N)
    if [ "$name" == "OpenMP" ]; then
        timeout 60 ./$executable --openmp < /dev/null > temp_output.log 2>&1
    else
        timeout 60 ./$executable < /dev/null > temp_output.log 2>&1
    fi
    exit_code=$?
    end_time=$(date +%s.%N)
    
    if [ $exit_code -eq 124 ]; then
        echo -e "${RED}TIMEOUT${NC}"
        echo "$name,$scene,$threads,$iteration,TIMEOUT,0,0" >> $RESULTS_FILE
        return
    elif [ $exit_code -ne 0 ]; then
        echo -e "${RED}FAILED${NC}"
        echo "$name,$scene,$threads,$iteration,FAILED,0,0" >> $RESULTS_FILE
        return
    fi
    
    # Calculate elapsed time
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # Get image dimensions (assuming 640x480 for simple, 800x600 for medium, 1280x720 for complex)
    case $scene in
        simple.txt)
            width=640; height=480 ;;
        medium.txt)
            width=800; height=600 ;;
        complex.txt)
            width=1280; height=720 ;;
        *)
            width=640; height=480 ;;
    esac
    
    # Calculate pixels per second
    pixels_per_sec=$(calc_pixels_per_second $elapsed $width $height)
    
    # Calculate speedup (will be computed later against serial baseline)
    echo "$name,$scene,$threads,$iteration,$elapsed,$pixels_per_sec,1.0" >> $RESULTS_FILE
    
    echo -e "${GREEN}${elapsed}s${NC} (${pixels_per_sec} pixels/s)"
}

# Function to compute statistics
compute_stats() {
    local impl=$1
    local scene=$2
    local threads=$3
    
    # Extract times for this configuration
    grep "^$impl,$scene,$threads," $RESULTS_FILE | cut -d',' -f5 | \
        awk '{sum+=$1; sumsq+=$1*$1; n++} 
             END {if(n>0) {avg=sum/n; std=sqrt(sumsq/n - avg*avg); 
                  printf "%.3f ± %.3f", avg, std} else print "N/A"}'
}

# Main benchmark loop
echo "Configuration:"
echo "  Scenes: ${SCENES[@]}"
echo "  Thread counts: ${THREAD_COUNTS[@]}"
echo "  Iterations per test: $ITERATIONS"
echo ""

for scene in "${SCENES[@]}"; do
    echo -e "${BLUE}=== Scene: $scene ===${NC}"
    
    # Check if scene file exists
    if [ ! -f "scenes/$scene" ]; then
        echo -e "  ${RED}Scene file not found${NC}"
        continue
    fi
    
    # Serial baseline (single thread)
    echo "Serial Implementation:"
    for iter in $(seq 1 $ITERATIONS); do
        run_benchmark "ray_serial" "$scene" 1 $iter "Serial"
    done
    serial_avg=$(compute_stats "Serial" "$scene" 1)
    echo "  Average: $serial_avg seconds"
    echo ""
    
    # OpenMP with different thread counts
    echo "OpenMP Implementation:"
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "  Testing with $threads threads:"
        for iter in $(seq 1 $ITERATIONS); do
            run_benchmark "ray_openmp" "$scene" $threads $iter "OpenMP"
        done
        omp_avg=$(compute_stats "OpenMP" "$scene" $threads)
        echo "  Average: $omp_avg seconds"
    done
    echo ""
    
    # CUDA GPU
    if [ -f "ray_cuda" ]; then
        echo "CUDA Implementation:"
        for iter in $(seq 1 $ITERATIONS); do
            run_benchmark "ray_cuda" "$scene" 1 $iter "CUDA"
        done
        cuda_avg=$(compute_stats "CUDA" "$scene" 1)
        echo "  Average: $cuda_avg seconds"
        echo ""
    fi
    
    # Hybrid
    if [ -f "ray_hybrid" ]; then
        echo "Hybrid Implementation:"
        for iter in $(seq 1 $ITERATIONS); do
            run_benchmark "ray_hybrid" "$scene" 4 $iter "Hybrid"
        done
        hybrid_avg=$(compute_stats "Hybrid" "$scene" 4)
        echo "  Average: $hybrid_avg seconds"
        echo ""
    fi
done

# Generate summary report
echo ""
echo "========================================="
echo "Performance Summary Report"
echo "========================================="
echo ""

# Function to generate summary table
generate_summary() {
    echo "Average Execution Times (seconds):"
    echo "-----------------------------------"
    printf "%-15s %-12s %-12s %-12s\n" "Implementation" "Simple" "Medium" "Complex"
    echo "-----------------------------------"
    
    # Serial
    printf "%-15s" "Serial"
    for scene in "${SCENES[@]}"; do
        avg=$(compute_stats "Serial" "$scene" 1)
        printf " %-12s" "$avg"
    done
    echo ""
    
    # OpenMP (best thread count)
    printf "%-15s" "OpenMP (4t)"
    for scene in "${SCENES[@]}"; do
        avg=$(compute_stats "OpenMP" "$scene" 4)
        printf " %-12s" "$avg"
    done
    echo ""
    
    # CUDA
    if [ -f "ray_cuda" ]; then
        printf "%-15s" "CUDA"
        for scene in "${SCENES[@]}"; do
            avg=$(compute_stats "CUDA" "$scene" 1)
            printf " %-12s" "$avg"
        done
        echo ""
    fi
    
    # Hybrid
    if [ -f "ray_hybrid" ]; then
        printf "%-15s" "Hybrid"
        for scene in "${SCENES[@]}"; do
            avg=$(compute_stats "Hybrid" "$scene" 4)
            printf " %-12s" "$avg"
        done
        echo ""
    fi
    
    echo "-----------------------------------"
}

generate_summary

# Calculate speedups
echo ""
echo "Speedup Analysis:"
echo "-----------------"

# Function to calculate and display speedup
calc_speedup() {
    local baseline_impl=$1
    local baseline_scene=$2
    local baseline_threads=$3
    local compare_impl=$4
    local compare_scene=$5
    local compare_threads=$6
    
    baseline=$(grep "^$baseline_impl,$baseline_scene,$baseline_threads," $RESULTS_FILE | \
               cut -d',' -f5 | awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')
    compare=$(grep "^$compare_impl,$compare_scene,$compare_threads," $RESULTS_FILE | \
              cut -d',' -f5 | awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')
    
    if [ "$baseline" != "0" ] && [ "$compare" != "0" ]; then
        speedup=$(echo "scale=2; $baseline / $compare" | bc)
        echo "$speedup"
    else
        echo "N/A"
    fi
}

# Display speedups for medium scene
echo "Speedups (Medium Scene):"
serial_time=$(calc_speedup "Serial" "medium.txt" 1 "Serial" "medium.txt" 1)
omp_speedup=$(calc_speedup "Serial" "medium.txt" 1 "OpenMP" "medium.txt" 4)
cuda_speedup=$(calc_speedup "Serial" "medium.txt" 1 "CUDA" "medium.txt" 1)
hybrid_speedup=$(calc_speedup "CUDA" "medium.txt" 1 "Hybrid" "medium.txt" 4)

echo "  OpenMP vs Serial:  ${omp_speedup}x"
if [ -f "ray_cuda" ]; then
    echo "  CUDA vs Serial:    ${cuda_speedup}x"
fi
if [ -f "ray_hybrid" ]; then
    echo "  Hybrid vs CUDA:    ${hybrid_speedup}x"
fi

# Check if requirements are met
echo ""
echo "Requirement Verification:"
echo "-------------------------"

if (( $(echo "$omp_speedup >= 2.5" | bc -l) )); then
    echo -e "  OpenMP speedup:  ${GREEN}✓ PASSED${NC} (${omp_speedup}x ≥ 2.5x)"
else
    echo -e "  OpenMP speedup:  ${RED}✗ FAILED${NC} (${omp_speedup}x < 2.5x)"
fi

if [ -f "ray_cuda" ]; then
    if (( $(echo "$cuda_speedup >= 10" | bc -l) )); then
        echo -e "  CUDA speedup:    ${GREEN}✓ PASSED${NC} (${cuda_speedup}x ≥ 10x)"
    else
        echo -e "  CUDA speedup:    ${RED}✗ FAILED${NC} (${cuda_speedup}x < 10x)"
    fi
fi

if [ -f "ray_hybrid" ]; then
    if (( $(echo "$hybrid_speedup >= 1.2" | bc -l) )); then
        echo -e "  Hybrid speedup:  ${GREEN}✓ PASSED${NC} (${hybrid_speedup}x ≥ 1.2x over GPU)"
    else
        echo -e "  Hybrid speedup:  ${RED}✗ FAILED${NC} (${hybrid_speedup}x < 1.2x over GPU)"
    fi
fi

# Save summary to file
echo ""
echo "Results saved to: $RESULTS_FILE"

# Create a plot script (optional, requires gnuplot)
if command -v gnuplot > /dev/null; then
    cat > "$OUTPUT_DIR/plot_${TIMESTAMP}.gnuplot" << EOF
set terminal png size 1200,800
set output '$OUTPUT_DIR/speedup_${TIMESTAMP}.png'
set title 'Ray Tracer Performance Comparison'
set xlabel 'Number of Threads'
set ylabel 'Speedup'
set grid
set key left top
plot '$RESULTS_FILE' using 3:7 with linespoints title 'OpenMP Speedup'
EOF
    
    echo "Gnuplot script created: $OUTPUT_DIR/plot_${TIMESTAMP}.gnuplot"
fi

# Cleanup
rm -f temp_output.log

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
