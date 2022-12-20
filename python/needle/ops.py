"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List, Union, Tuple
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy
import numpy as np

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)

        if len(out_grad) > 1:
            return tuple([out_grad[i] for i in range(len(out_grad))])
        else:
            return out_grad[0]


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        return self.scalar * out_grad * node.inputs[0]**(self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


def square(a):
    return PowerScalar(2)(a)


def sqrt(a):
    return PowerScalar(0.5)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        divisible, divisor = node.inputs
        return out_grad / divisor, \
                out_grad * (-divisible) / divisor**2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if not (axes is None or len(axes) == 2):
            raise ValueError('Wrong new axes')
        
        self.axes = (-2, -1) if axes is None else axes

    def compute(self, a):
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Permute(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if not (axes is None or set(axes) == set(range(len(axes)))):
            raise ValueError('Wrong new axes')

        self.axes = list(range(axes))[::-1] if axes is None else axes

    def compute(self, a):
        if isinstance(a, np.ndarray):
            return np.transpose(a, self.axes)
        elif isinstance(a, NDArray):
            return a.permute(self.axes)
        else:
            raise ValueError(f"Expected np.ndarray or NDArray, got {type(a)}")

    def gradient(self, out_grad, node):
        reversed_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            reversed_axes[ax] = i

        return permute(out_grad, tuple(reversed_axes))


def permute(a, axes=None):
    return Permute(axes)(a)        


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        inp_shape = node.inputs[0].shape
        return out_grad.reshape(inp_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        
    @staticmethod
    def find_first_occ(arr: tuple, subarr: tuple) -> List[bool]:
        mask = [False] * len(arr)
        mask_start = 0

        for start in range(len(arr) - len(subarr) + 1):
            for i in range(start, start+len(subarr)):
                if arr[i] == subarr[i-start] or subarr[i] == 1:
                    continue
                else:
                    break
            else:
                mask_start = start
                break
                
        if mask_start is not None:
            for i in range(mask_start, mask_start+len(subarr)):
                if subarr[i-mask_start] != 1:
                    mask[i] = True
        
        return mask

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        inp_shape = node.inputs[0].shape
        
        if inp_shape == out_grad.shape:
            return out_grad
        elif node.inputs[0].size == 1:
            return out_grad.sum().reshape(inp_shape)
        
        shape_mask = BroadcastTo.find_first_occ(out_grad.shape, 
                                                inp_shape)
        rev_shape_mask = [not bool_val for bool_val in shape_mask]

        dim_to_cut = np.arange(len(out_grad.shape), dtype=int)
        dim_to_cut = tuple(dim_to_cut[rev_shape_mask])

        return out_grad.sum(axes=dim_to_cut).reshape(inp_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None, keepdims: bool = False):
        if isinstance(axes, int):
            axes = axes,

        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        result = a.sum(self.axes)

        if self.keepdims:
            outp_shape = get_unsq_outp_shape(list(a.shape), self.axes)
            return result.reshape(outp_shape)
        else:
            return result

    def gradient(self, out_grad, node):
        inp_shape = list(node.inputs[0].shape)
        unsq_outp_shape = get_unsq_outp_shape(inp_shape, self.axes)

        return broadcast_to(out_grad.reshape(unsq_outp_shape), inp_shape)


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class Mean(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None, keepdims: bool = False):
        if isinstance(axes, int):
            axes = axes,

        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        result = array_api.mean(a, self.axes)

        if self.keepdims:
            outp_shape = get_unsq_outp_shape(list(a.shape), self.axes)
            return result.reshape(outp_shape)
        else:
            return result

    def gradient(self, out_grad, node):
        size_diff = node.inputs[0].size / out_grad.size
        sum_grad = Summation(self.axes, self.keepdims).gradient(out_grad, node)
        return sum_grad / size_diff


def mean(a, axes=None, keepdims=False):
    return Mean(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        left_mat, right_mat = node.inputs
        left_mat_tr = left_mat.transpose()
        right_mat_tr = right_mat.transpose()

        left_grad = out_grad @ right_mat_tr
        right_grad = left_mat_tr @ out_grad

        kdim_to_reduct_a = left_grad.ndim - left_mat.ndim
        kdim_to_reduct_b = right_grad.ndim - right_mat.ndim

        if kdim_to_reduct_a > 0:
            left_grad = left_grad.sum(tuple(range(kdim_to_reduct_a)))

        if kdim_to_reduct_b > 0:
            right_grad = right_grad.sum(tuple(range(kdim_to_reduct_b)))

        return left_grad, right_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        bool_mask = a > 0
        return a * bool_mask

    def gradient(self, out_grad, node):
        node.inputs[0].realize_cached_data()
        layer_mask = Tensor(node.inputs[0].cached_data > 0, 
                            device=out_grad.device)
        return out_grad * layer_mask


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None):
        if isinstance(axes, int):
            axes = axes,
        
        self.axes = axes

    def broadcasted_max(self, inp):
        # analog of np.max(arr, self.axes, keepdims=True)
        unsq_outp_shape = get_unsq_outp_shape(list(inp.shape), self.axes)

        is_tensor = isinstance(inp, Tensor)
        inp_data = inp
        if is_tensor:
            inp.realize_cached_data()
            inp_data = inp.cached_data

        inp_max = inp_data.max(self.axes)
        broadcasted_max = inp_max.reshape(unsq_outp_shape)
        broadcasted_max = array_api.broadcast_to(broadcasted_max, inp.shape)

        return Tensor(broadcasted_max, device=inp.device) if is_tensor \
            else broadcasted_max

    def compute(self, Z: numpy.ndarray):
        max_z = Z.max(self.axes)
        broadcasted_max_z = self.broadcasted_max(Z)

        return array_api.log(
                    array_api.sum(
                        array_api.exp(Z - broadcasted_max_z), 
                        self.axes
                    )
                ) + max_z


    def gradient(self, out_grad, node):
        inp = node.inputs[0]
        inp_shape = list(inp.shape)
        unsq_outp_shape = get_unsq_outp_shape(inp_shape, self.axes)

        broadcasted_max = self.broadcasted_max(inp)

        # Calculating numerically stable Softmax
        inp -= broadcasted_max
        exp_inp = exp(inp)
        sum_exp = exp_inp.sum(self.axes).reshape(unsq_outp_shape)
        sum_exp = sum_exp.broadcast_to(inp_shape)
        softmax = exp_inp / sum_exp

        broadcasted_out_grad = broadcast_to(out_grad.reshape(unsq_outp_shape), 
                                            inp_shape)
        return softmax * broadcasted_out_grad


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Variation(TensorOp):
    def __init__(self, axes: Optional[Union[tuple, int]] = None, keepdims: bool = False):
        if isinstance(axes, int):
            axes = axes,
        
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        result = array_api.var(a, self.axes)

        if self.keepdims:
            outp_shape = get_unsq_outp_shape(list(a.shape), self.axes)
            return result.reshape(outp_shape)
        else:
            return result

    def gradient(self, out_grad, node):
        inp = node.inputs[0]
        inp_shape = list(inp.shape)

        inp_mean = inp.mean(self.axes, keepdims=True)
        inp_mean = inp_mean.broadcast_to(inp_shape)

        reduced_size = inp.size if self.axes is None \
                        else numpy.product([inp.shape[ax] for ax in self.axes])
        grad_coeff = 2. / reduced_size

        broadcasted_grad = Summation(self.axes, self.keepdims).gradient(out_grad, node)

        return grad_coeff * broadcasted_grad * (inp - inp_mean)


def variation(a, axes=None, keepdims=False):
    return Variation(axes, keepdims)(a)


class Tanh(TensorOp):
    def compute(self, a):
        if isinstance(a, Tensor):
            return 1 - 2 / (exp(2 * a) + 1)
        else:
            return 1 - 2 / (array_api.exp(2 * a) + 1)

    def gradient(self, out_grad, node):
        # tanh'(x) = 1 - tanh^2(x)
        inp = node.inputs[0]
        return out_grad * (1 - self.compute(inp)**2)


def tanh(a):
    return Tanh()(a)

class Sin(TensorOp):
    def compute(self, a):
        return array_api.sin(a)

    def gradient(self, out_grad, node):
        # tanh'(x) = 1 - tanh^2(x)
        inp = node.inputs[0]
        return out_grad * cos(inp)


def sin(a):
    return Sin()(a)

class Cos(TensorOp):
    def compute(self, a):
        return array_api.cos(a)

    def gradient(self, out_grad, node):
        # tanh'(x) = 1 - tanh^2(x)
        inp = node.inputs[0]
        return out_grad * (-sin(inp))


def cos(a):
    return Cos()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        elem_shape = args[0].shape
        elem_device = args[0].device
        
        result_shape = list(elem_shape)
        result_shape.insert(self.axis, len(args))
        result_shape = tuple(result_shape)
        result = array_api.zeros(result_shape, device=elem_device)

        for i in range(len(args)):
            result[make_slice(result_shape, self.axis, i)] = args[i]

        return result

    def gradient(self, out_grad, node):
        splitted_grad = split(out_grad, self.axis)
        return splitted_grad


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        result = []

        for i in range(A.shape[self.axis]):
            elem = array_api.squeeze(
                A[make_slice(A.shape, self.axis, i)],
                self.axis
            )
            result.append(elem)

        return result

    def gradient(self, out_grad, node):
        stacked_tensor = stack(out_grad, self.axis)
        return stacked_tensor


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if not (axes is None or isinstance(axes, tuple)):
            raise ValueError(f'Expected type(axes) = tuple, got {type(axes)}')
        
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        result_data = np.flip(out_grad.numpy(), self.axes)
        return Tensor(result_data, device=out_grad.device)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if max(self.axes) >= a.ndim:
            return a

        slices_arr = [slice(sh) for sh in a.shape]
        result_shape = list(a.shape)

        for ax in self.axes:
            result_shape[ax] *= (self.dilation + 1)
            slices_arr[ax] = slice(0, result_shape[ax], self.dilation + 1)

        result = array_api.zeros(result_shape, device=a.device)
        result[tuple(slices_arr)] = a

        return result

    def gradient(self, out_grad, node):
        result = undilate(out_grad, self.axes, self.dilation)
        return result


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if max(self.axes) >= a.ndim:
            return a

        slices_arr = [slice(sh) for sh in a.shape]
        
        for ax in self.axes:
            slices_arr[ax] = slice(0, a.shape[ax], self.dilation + 1)

        return a[tuple(slices_arr)]

    def gradient(self, out_grad, node):
        result = dilate(out_grad, self.axes, self.dilation)
        return result


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0, 
                    dilation: Optional[int] = 1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def compute(self, A, B):
        padding_arr = ((0,)*2, (self.padding,)*2, (self.padding,)*2, (0,)*2)
        A_padded = array_api.pad(A, padding_arr)

        batch_size, height, width, in_ch = A_padded.shape
        kernel, _, _, out_ch = B.shape
        batch_s, height_s, width_s, in_ch_s = A_padded.strides

        inner_dim = kernel**2 * in_ch
        modified_kernel = kernel + (kernel - 1) * (self.dilation - 1)
        new_shape = (
            batch_size, 
            len(range(0, height-modified_kernel+1, self.stride)), 
            len(range(0, width-modified_kernel+1, self.stride)), 
            kernel, 
            kernel, 
            in_ch
        )
        new_strides = (
            batch_s, 
            height_s * self.stride, 
            width_s * self.stride, 
            height_s * self.dilation, 
            width_s * self.dilation, 
            in_ch_s
        )

        if isinstance(A, np.ndarray):
            A_modified = np.lib.stride_tricks.as_strided(
                A_padded, 
                shape=new_shape,
                strides=new_strides
            ).reshape(-1, inner_dim)
        elif isinstance(A, NDArray):
            A_modified = NDArray.make(
                new_shape,
                new_strides,
                A.device,
                A_padded._handle,
                A_padded._offset
            ).compact().reshape((np.prod(new_shape) // inner_dim, inner_dim))
        else:
            raise ValueError(f"Expected np.ndarray or NDArray, got {type(A)}")

        B_modified = B.reshape((inner_dim, out_ch))
        result = A_modified @ B_modified
        
        return result.reshape((
            batch_size, 
            new_shape[1], 
            new_shape[2], 
            out_ch
        ))

    def gradient(self, out_grad, node):
        x, w = node.inputs
        kernel = w.shape[0]
        # Calculate gradient for X
        # X.grad = out_grad @ W.T \approx Conv(\approx out_grad, \approx W)
        # Conv.stride = 1
        # Conv.padding = len(range(0, 2K-2, self.stride)) - self.padding
        # \approx out_grad = out_grad + Dilate(axes=(1, 2), self.stride-1)
        # \approx W = W + Flip(axes=(0, 1)) + Transpose(axes=(2, 3))
        conv_padding = kernel - 1 - self.padding
        modified_out_grad = dilate(out_grad, (1, 2), self.stride-1)
        modified_w = transpose(flip(w, (0, 1)), (2, 3))

        if isinstance(w.realize_cached_data(), NDArray):
            modified_out_grad = Tensor(
                modified_out_grad.cached_data.compact(),
                device=out_grad.device
            )
            modified_w = Tensor(
                modified_w.cached_data.compact(),
                device=out_grad.device
            )

        x_grad = conv(modified_out_grad, modified_w, 1, max(conv_padding, 0))
        if conv_padding < 0:
            x_grad = x_grad[:, -conv_padding:conv_padding, -conv_padding:conv_padding, :]

        # Calculate gradient for W
        # W,grad = X.T @ out_grad \approx Conv(\approx X, \approx out_grad)
        # Conv.stride = 1
        # Conv.padding = self.padding
        # \approx X = X + Transpose(axes=(0, 3)) (BHWC ==> CHWB)
        # \approx out_grad = out_grad + Permute(1, 2, 0, 3) (BH'W'O ==> H'W'BO) + 
        #                    + Dilate(axes=(0, 1), self.stride-1)
        # w_grad = w_grad + Permute(1, 2, 0, 3) (CKKO ==> KKCO)
        permuted_x = transpose(x, (0, 3))
        permuted_out_grad = permute(out_grad, (1, 2, 0, 3))
        permuted_out_grad = dilate(permuted_out_grad, (0, 1), self.stride-1)

        if isinstance(w.realize_cached_data(), NDArray):
            permuted_x = Tensor(
                permuted_x.cached_data.compact(),
                device=out_grad.device
            )
            permuted_out_grad = Tensor(
                permuted_out_grad.cached_data.compact(),
                device=out_grad.device
            )

        w_grad = conv(permuted_x, permuted_out_grad, 1, self.padding)
        w_grad = permute(w_grad, (1, 2, 0, 3))
        
        return x_grad, w_grad


def conv(a, b, stride=1, padding=1, dilation=1):
    return Conv(stride, padding, dilation)(a, b)

# class Noise(TensorOp):
#     def __init__(self, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
#         self.sqrt_alpha_cumprod = sqrt_alpha_cumprod
#         self.sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod


#     def compute(self, a):
#         '''
#         Перегоняет исходное изображение на зашумленный шаг t, отсюда и пересчитанные альфы
#         '''
#         noise = init.randn(a.shape, device=a.device)
#         return self.sqrt_alpha_cumprod * a + self.sqrt_one_minus_alpha_cumprod * noise, noise

#     def gradient(self, out_grad, node):
#         raise NotImplementedError

# def noise(a, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
#     return Noise(sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod)(a)

class Abs(TensorOp):
    def compute(self, X: NDArray):
        return array_api.maximum(X, -X)

    def gradient(self, out_grad, node):
        X = node.inputs[0]
        mask = -2 * (X < 0) + 1
        return out_grad * mask

def abs(a):
    return Abs()(a)

# Helper functions
def get_unsq_outp_shape(inp_shape: List[int], axes: Optional[tuple] = None):
    if isinstance(inp_shape, tuple):
        inp_shape = list(inp_shape)
        
    outp_shape = inp_shape.copy()
    
    if axes is None:
        outp_shape[:] = [1] * len(inp_shape)
    else:
        for ax in axes:
            outp_shape[ax] = 1

    return outp_shape


def make_slice(shape: tuple, axis: int, ind: int) -> Tuple[slice]:
    slices = [slice(sh) for sh in shape]
    slices[axis] = slice(ind, ind + 1, 1)
    return tuple(slices)
