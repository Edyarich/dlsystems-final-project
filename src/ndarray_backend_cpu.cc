#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
typedef int32_t shape_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}


struct MultidimCounter {
  MultidimCounter(const std::vector<shape_t>& border) {
    std::vector<shape_t> temp(border.size(), 0);
    this->counter = temp;
    this->border = border;
  }

  bool increment(size_t step = 1) {
    int64_t ind = counter.size() - 1;

    while (ind >= 0) {
      if (counter[ind] + step < border[ind]) {
        counter[ind] += step;

        for (size_t j = ind + 1; j < counter.size(); ++j) {
          counter[j] = 0;
        }
        break;
      } else {
        ind -= 1;
      }
    }
    
    return ind >= 0;
  }

  int64_t total(const std::vector<shape_t>& strides) {
    int64_t idx = 0;

    for (int i = 0; i < counter.size(); ++i) {
      idx += strides[i] * counter[i];
    }

    return idx;
  }

  friend std::ostream& operator << (std::ostream& out, const MultidimCounter& cnter){
    for (auto count: cnter.counter) {
      out << count << ' ';
    }
    out << std::endl;
    return out;
  }

  std::vector<shape_t> counter;
  std::vector<shape_t> border;
};


void Compact(const AlignedArray& a, AlignedArray* out, std::vector<shape_t> shape,
             std::vector<shape_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */

  if (shape.size() == 0) {
    return;
  }

  auto multidim_counter = MultidimCounter(shape);
  size_t out_counter = 0;

  do {
    out->ptr[out_counter++] = a.ptr[offset + multidim_counter.total(strides)];
  } while (multidim_counter.increment() && out_counter < out->size);
}


void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<shape_t> shape,
                  std::vector<shape_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
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

  auto multidim_counter = MultidimCounter(shape);
  size_t a_counter = 0;

  do {
    out->ptr[offset + multidim_counter.total(strides)] = a.ptr[a_counter++];
  } while (multidim_counter.increment() && a_counter < a.size);
}


void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<shape_t> shape,
                   std::vector<shape_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
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
    
  auto multidim_counter = MultidimCounter(shape);
  size_t k_assigned = 0;
  
  do {
    ++k_assigned;
    out->ptr[offset + multidim_counter.total(strides)] = val;
  } while (multidim_counter.increment() && k_assigned < size);
}


void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
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

template <typename UnaryOp>
void PerformUnaryOp(const AlignedArray& a, AlignedArray* out) {
  auto func = UnaryOp();
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = func(a.ptr[i]);
  }
}


template <typename BinaryOp>
void PerformBinaryOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  auto func = BinaryOp();
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = func(a.ptr[i], b.ptr[i]);
  }
}


template <typename ScalarOp>
void PerformScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  auto func = ScalarOp(val);
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = func(a.ptr[i]);
  }
}


void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  struct Mul {
    scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
      return first_elem * second_elem;
    }
  };

  PerformBinaryOp<Mul>(a, b, out);
}


void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  struct Mul {
    Mul(scalar_t val) {
      this->val = val;
    }

    scalar_t operator () (const scalar_t& elem) {
      return elem * val;
    }

    scalar_t val;
  };

  PerformScalarOp<Mul>(a, val, out);
}


void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  struct Div {
    scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
        return first_elem * 1.0f / second_elem;
    }
  };

  PerformBinaryOp<Div>(a, b, out);
}


void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  struct Div {
    Div(scalar_t val) {
      this->val = val;
    }

    scalar_t operator () (const scalar_t& elem) {
      return elem * 1.0f / val;
    }

    scalar_t val;
  };

  PerformScalarOp<Div>(a, val, out);
}


void ScalarPower(const AlignedArray& a, scalar_t degree, AlignedArray* out) {
  struct Power {
    Power(scalar_t deg) {
      this->deg = deg;
    }

    float operator () (const scalar_t& elem) {
      return std::pow(elem, deg);
    }

    scalar_t deg;
  };

  PerformScalarOp<Power>(a, degree, out);
}


void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  struct Max {
    scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
      return std::max(first_elem, second_elem);
    }
  };

  PerformBinaryOp<Max>(a, b, out);
}


void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  struct Max {
    Max(scalar_t val) {
      this->val = val;
    }

    scalar_t operator () (const scalar_t& elem) {
      return std::max(elem, val);
    }

    scalar_t val;
  };

  PerformScalarOp<Max>(a, val, out);
}


void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  struct Eq {
    scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
      return first_elem == second_elem;
    }
  };

  PerformBinaryOp<Eq>(a, b, out);
}


void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  struct Eq {
    Eq(scalar_t val) {
      this->val = val;
    }

    scalar_t operator () (const scalar_t& elem) {
      return elem == val;
    }

    scalar_t val;
  };

  PerformScalarOp<Eq>(a, val, out);
}


void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  struct Geq {
    scalar_t operator () (const scalar_t& first_elem, const scalar_t& second_elem) {
      return first_elem - second_elem >= 0;
    }
  };

  PerformBinaryOp<Geq>(a, b, out);
}


void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  struct Geq {
    Geq(scalar_t val) {
      this->val = val;
    }

    scalar_t operator () (const scalar_t& elem) {
      return elem - val >= 0;
    }

    scalar_t val;
  };

  PerformScalarOp<Geq>(a, val, out);
}


void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  struct Log {
    scalar_t operator () (const scalar_t& elem) {
      return std::log(elem);
    }
  };

  PerformUnaryOp<Log>(a, out);
}


void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  struct Exp {
    scalar_t operator () (const scalar_t& elem) {
      return std::exp(elem);
    }
  };

  PerformUnaryOp<Exp>(a, out);
}

void EwiseSin(const AlignedArray& a, AlignedArray* out) {
  struct Sin {
    scalar_t operator () (const scalar_t& elem) {
      return std::sin(elem);
    }
  };

  PerformUnaryOp<Sin>(a, out);
}

void EwiseCos(const AlignedArray& a, AlignedArray* out) {
  struct Cos {
    scalar_t operator () (const scalar_t& elem) {
      return std::cos(elem);
    }
  };

  PerformUnaryOp<Cos>(a, out);
}


void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  struct Tanh {
    scalar_t operator () (const scalar_t& elem) {
      return std::tanh(elem);
    }
  };

  PerformUnaryOp<Tanh>(a, out);
}


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < p; ++j) {
      scalar_t res = 0;
      for (uint32_t k = 0; k < n; ++k) {
        res += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
      out->ptr[i * p + j] = res;
    }
  }
}


inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  for (uint32_t i = 0; i < TILE; ++i) {
    for (uint32_t j = 0; j < TILE; ++j) {
      scalar_t dot_prod = 0;
      for (uint32_t k = 0; k < TILE; ++k) {
        dot_prod += a[i * TILE + k] * b[k * TILE + j];
      }
      out[i * TILE + j] += dot_prod;
    }
  }
}


void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  std::memset(out->ptr, 0, sizeof(scalar_t)*m*p);

  for (uint32_t i = 0; i < m / TILE; ++i) {
    for (uint32_t j = 0; j < p / TILE; ++j) {
      for (uint32_t k = 0; k < n / TILE; ++k) {
        AlignedDot(
          a.ptr + i * n * TILE + k * TILE * TILE, 
          b.ptr + k * p * TILE + j * TILE * TILE, 
          (out->ptr) + i * p * TILE + j * TILE * TILE
        );
      }
    }
  }
}


void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  for (size_t i = 0; i < out->size; ++i) {
    scalar_t max_value = a.ptr[i*reduce_size];
    for (size_t j = 1; j < reduce_size; ++j) {
      max_value = std::max(max_value, a.ptr[i*reduce_size+j]);
    }
    out->ptr[i] = max_value;
  }
}


void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  for (size_t i = 0; i < out->size; ++i) {
    scalar_t total_sum = 0;
    for (size_t j = 0; j < reduce_size; ++j) {
      total_sum += a.ptr[i*reduce_size+j];
    }
    out->ptr[i] = total_sum;
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("ewise_sin", EwiseSin);
  m.def("ewise_cos", EwiseCos);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
