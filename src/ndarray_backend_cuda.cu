#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cassert>

#include "math.h"

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
const int MAX_X_BLOCKS = 65535;
const int MAX_Y_BLOCKS = 65535;
const int BLOCK_X_SIZE = 16;
const int BLOCK_Y_SIZE = BASE_THREAD_NUM / BLOCK_X_SIZE;

#define TILE 4
typedef float scalar_t;
typedef int32_t shape_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  shape_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<shape_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Utility function to convert contiguous index i to memory location from strides
struct MultidimCounter {
  __device__ MultidimCounter(CudaVec border) {
    this->border = border;
    this->counter.size = border.size;
    this->row_strides.size = border.size;

    size_t total_size = 1;

    for (int i = 0; i < border.size; ++i) {
      this->counter.data[i] = 0;
      total_size *= border.data[i];
    }

    for (int i = 0; i < border.size; ++i) {
      total_size /= border.data[i];
      this->row_strides.data[i] = total_size;
    }
  }

  __device__ bool single_increment() {
    int64_t ind = counter.size - 1;

    while (ind >= 0) {
      if (counter.data[ind] + 1 < border.data[ind]) {
        counter.data[ind] += 1;

        for (size_t j = ind + 1; j < counter.size; ++j) {
          counter.data[j] = 0;
        }
        break;
      } else {
        ind -= 1;
      }
    }
    
    return ind >= 0;
  }

  __device__ bool increment(size_t step = 1) {
    sum += step;

    if (step == 1) {
      return single_increment();
    } else {
      int64_t residual = sum;

      for (int i = 0; i < counter.size; ++i) {
        uint32_t temp = residual / row_strides.data[i];
        residual -= temp * row_strides.data[i];

        if (temp >= border.data[i]) {
          return false;
        } else {
          counter.data[i] = temp;
        }
      }

      return true;
    }
  }

  __device__ int64_t total(CudaVec strides) {
    int64_t idx = 0;

    for (int i = 0; i < counter.size; ++i) {
      idx += strides.data[i] * counter.data[i];
    }

    return idx;
  }

  CudaVec counter;
  CudaVec border;
  CudaVec row_strides;
  size_t sum = 0;
};


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  auto multidim_counter = MultidimCounter(shape);
  bool is_successful = multidim_counter.increment(gid);

  for (size_t i = gid; i < size; i += step) {
    if (is_successful == false) {
      return;
    }
    out[i] = a[offset + multidim_counter.total(strides)];
    is_successful = multidim_counter.increment(step);
  }
}

void Compact(const CudaArray& a, CudaArray* out, const std::vector<shape_t>& shape,
             const std::vector<shape_t>& strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  if (shape.size() == 0) {
    return;
  }

  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  auto multidim_counter = MultidimCounter(shape);
  bool is_successful = multidim_counter.increment(gid);

  for (size_t i = gid; i < size; i += step) {
    if (is_successful == false) {
      return;
    }
    out[offset + multidim_counter.total(strides)] = a[i];
    is_successful = multidim_counter.increment(step);
  }
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, const std::vector<shape_t>& shape,
                  const std::vector<shape_t>& strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  if (shape.size() == 0) {
    return;
  }

  CudaDims dim = CudaOneDim(out->size);
  EwiseSetKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                          VecToCuda(strides), offset);
}


__global__ void ScalarSetKernel(scalar_t* out, scalar_t val, size_t size, CudaVec shape,
                                CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  auto multidim_counter = MultidimCounter(shape);
  bool is_successful = multidim_counter.increment(gid);

  for (size_t i = gid; i < size; i += step) {
    if (is_successful == false) {
      return;
    }
    out[offset + multidim_counter.total(strides)] = val;
    is_successful = multidim_counter.increment(step);
  }
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, const std::vector<shape_t>& shape,
                   const std::vector<shape_t>& strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  if (shape.size() == 0) {
    return;
  }

  CudaDims dim = CudaOneDim(size);
  ScalarSetKernel<<<dim.grid, dim.block>>>(out->ptr, val, size, VecToCuda(shape),
                                          VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

// Template kernels
template <typename UnaryOp>
__global__ void UnaryOpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  auto func = UnaryOp();
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (int i = gid; i < size; i += step) {
    out[i] = func(a[i]);
  }
}

template <typename BinaryOp>
__global__ void BinaryOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  auto func = BinaryOp();
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (int i = gid; i < size; i += step) {
    out[i] = func(a[i], b[i]);
  }
}

template <typename ScalarOp>
__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  auto func = ScalarOp(val);
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;

  for (int i = gid; i < size; i += step) {
    out[i] = func(a[i]);
  }
}

// Functors
struct Mul {
  __device__ scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
    return first_elem * second_elem;
  }
};


struct EMul {
  __device__ EMul(scalar_t val) {
    this->val = val;
  }

  __device__ scalar_t operator () (const scalar_t& elem) {
    return elem * val;
  }

  scalar_t val;
};


struct Div {
  __device__ scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
      return first_elem * 1.0f / second_elem;
  }
};


struct EDiv {
  __device__ EDiv(scalar_t val) {
    this->val = val;
  }

  __device__ scalar_t operator () (const scalar_t& elem) {
    return elem * 1.0f / val;
  }

  scalar_t val;
};


struct Power {
  __device__ Power(scalar_t deg) {
    this->deg = deg;
  }

  __device__ float operator () (const scalar_t& elem) {
    return std::pow(elem, deg);
  }

  scalar_t deg;
};


struct Max {
  __device__ scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
    return max(first_elem, second_elem);
  }
};


struct EMax {
  __device__ EMax(scalar_t val) {
    this->val = val;
  }

  __device__ scalar_t operator () (const scalar_t& elem) {
    return max(elem, val);
  }

  scalar_t val;
};


struct Eq {
  __device__ scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
    return first_elem == second_elem;
  }
};


struct EEq {
  __device__ EEq(scalar_t val) {
    this->val = val;
  }

  __device__ scalar_t operator () (const scalar_t& elem) {
    return elem == val;
  }

  scalar_t val;
};


struct Geq {
  __device__ scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
    return first_elem - second_elem >= 0;
  }
};


struct EGeq {
  __device__ EGeq(scalar_t val) {
    this->val = val;
  }

  __device__ scalar_t operator () (const scalar_t& elem) {
    return elem - val >= 0;
  }

  scalar_t val;
};


struct Log {
  __device__ scalar_t operator () (const scalar_t& elem) {
    return log(elem);
  }
};


struct Exp {
  __device__ scalar_t operator () (const scalar_t& elem) {
    return std::exp(elem);
  }
};

struct Sin {
  __device__ scalar_t operator () (const scalar_t& elem) {
    return std::sin(elem);
  }
};

struct Cos {
  __device__ scalar_t operator () (const scalar_t& elem) {
    return std::cos(elem);
  }
};

struct Tanh {
  __device__ scalar_t operator () (const scalar_t& elem) {
    return std::tanh(elem);
  }
};

// The main part
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  BinaryOpKernel<Mul><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<EMul><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  BinaryOpKernel<Div><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<EDiv><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


void ScalarPower(const CudaArray& a, scalar_t degree, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<Power><<<dim.grid, dim.block>>>(a.ptr, degree, out->ptr, out->size);
}


void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  BinaryOpKernel<Max><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<EMax><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  BinaryOpKernel<Eq><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<EEq><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  BinaryOpKernel<Geq><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<EGeq><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<Log><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<Exp><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

void EwiseSin(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<Sin><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

void EwiseCos(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<Cos><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<Tanh><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
__global__ void KernelMatmul(int heightA, int widthA, int widthB,
                          float* matrixA, float* matrixB, float* matrixResult) {
  extern __shared__ float sh_data[];
  float* A_window = sh_data; // blockDim.x * blockDim.y
  float* B_window = (float*)&A_window[blockDim.x * blockDim.y]; // blockDim.y * blockDim.x

  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  int widthA_blocks = (widthA + blockDim.x - 1) / blockDim.x;
  float thread_sum = 0.0;

  for (int aBlockIdx = 0; aBlockIdx < widthA_blocks; ++aBlockIdx) {
    size_t curr_ind_a = aBlockIdx * blockDim.x + threadIdx.x;
    size_t a_idx = row * widthA + curr_ind_a;
    int a_window_ind = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < heightA && curr_ind_a < widthA) {
      A_window[a_window_ind] = matrixA[a_idx];
    } else {
      A_window[a_window_ind] = 0.0f;
    }

    size_t curr_ind_b = aBlockIdx * blockDim.x + threadIdx.y;
    size_t b_idx = curr_ind_b * widthB + col;
    int b_window_ind = threadIdx.y * blockDim.x + threadIdx.x;

    if (curr_ind_b < widthA && col < widthB) {
      B_window[b_window_ind] = matrixB[b_idx];
    } else {
      B_window[b_window_ind] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < blockDim.y; ++k) {
      thread_sum += A_window[threadIdx.y * blockDim.x + k] *
                    B_window[k * blockDim.x + threadIdx.x];
    }
    
    __syncthreads();
  }

  size_t res_ind = row * widthB + col;

  if (row < heightA && col < widthB) {
    matrixResult[res_ind] = thread_sum;
  }
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  int grid_x_size = fmin(MAX_X_BLOCKS, (P + BLOCK_X_SIZE - 1) / BLOCK_X_SIZE);
  int grid_y_size = fmin(MAX_Y_BLOCKS, (M + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE);
  int blocksize_in_bytes = BLOCK_X_SIZE * BLOCK_Y_SIZE * ELEM_SIZE;
  int sh_memory_size = 2 * blocksize_in_bytes;

  CudaDims dim;
  dim.block = dim3(BLOCK_X_SIZE, BLOCK_Y_SIZE, 1);
  dim.grid = dim3(grid_x_size, grid_y_size, 1);

  KernelMatmul<<<dim.grid, dim.block, sh_memory_size>>>(M, N, P, a.ptr, b.ptr, out->ptr);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void KernelMax(const float* a, float* out, size_t out_size, size_t reduce_size) {
  int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
  int row_step = blockDim.y * gridDim.y;

  for (int row = tid_y; row < out_size; row += row_step)  {
    float max_val = -INFINITY;

    for (int j = 0; j < reduce_size; ++j) {
      max_val = max(a[row * reduce_size + j], max_val);
    }

    out[row] = max_val;
  }
}


void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  int grid_y_size = fmin(MAX_Y_BLOCKS, (out->size + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE);

  CudaDims dim;
  dim.block = dim3(1, BLOCK_Y_SIZE, 1);
  dim.grid = dim3(1, grid_y_size, 1);

  KernelMax<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}


__global__ void KernelSum(const float* a, float* out, size_t out_size, size_t reduce_size) {
  int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
  int row_step = blockDim.y * gridDim.y;

  for (int row = tid_y; row < out_size; row += row_step)  {
    float sum = 0;

    for (int j = 0; j < reduce_size; ++j) {
      sum += a[row * reduce_size + j];
    }

    out[row] = sum;
  }
}


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  int grid_y_size = fmin(MAX_Y_BLOCKS, (out->size + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE);

  CudaDims dim;
  dim.block = dim3(1, BLOCK_Y_SIZE, 1);
  dim.grid = dim3(1, grid_y_size, 1);

  KernelSum<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}