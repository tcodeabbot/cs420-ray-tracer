#ifndef CUDA_FIX_H
#define CUDA_FIX_H

// CUDA 13.0 compatibility fix
// Define missing CUDART constants before including CUDA headers
#define CUDART_PI              3.1415926535897931e+0
#define CUDART_THIRD           3.3333333333333333e-1
#define CUDART_SQRT_HALF_HI    7.0710678118654757e-1
#define CUDART_SQRT_HALF_LO   (-1.7484555327724691e-17)

#endif // CUDA_FIX_H
