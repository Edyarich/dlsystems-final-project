"""The module.
"""
from typing import List, Optional
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(
                in_features,
                out_features,
                device=device,
                dtype=dtype
            )
        )
        self.has_bias = bias

        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    out_features,
                    1,
                    device=device,
                    dtype=dtype
                ).transpose()
            )

    def forward(self, X: Tensor) -> Tensor:
        batch_size = X.shape[0]
        logits = X @ self.weight

        if self.has_bias:
            reshaped_bias = ops.broadcast_to(self.bias,
                                            (batch_size, self.out_features))
            logits += reshaped_bias

        return logits


class Flatten(Module):
    def forward(self, X):
        batch_size = X.shape[0]
        return X.reshape((batch_size, X.size // batch_size))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * Tensor(x.numpy() > 0, device=x.device)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        inp = x

        for module in self.modules:
            outp = module(inp)
            inp = outp

        return outp


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        num_classes = logits.shape[-1]

        log_sum_exp = ops.logsumexp(logits, -1)
        onehot_y = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        y_logits = ops.summation(logits * onehot_y, -1)

        return ops.mean(log_sum_exp - y_logits)


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            reduced_dims = 0,
            mean = ops.mean(x, reduced_dims, False)
            broad_mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ops.variation(x, reduced_dims, False)
            broad_var = var.reshape((1, self.dim)).broadcast_to(x.shape)
            normalized_x = (x - broad_mean) / ops.sqrt(broad_var + self.eps)

            unsq_param_shape = [1] * len(x.shape)
            unsq_param_shape[-1] = self.dim
            weight = self.weight.reshape(unsq_param_shape).broadcast_to(x.shape)
            bias = self.bias.reshape(unsq_param_shape).broadcast_to(x.shape)

            self.running_mean *= (1 - self.momentum)
            self.running_mean += self.momentum * mean.data
            self.running_var *= (1 - self.momentum)
            self.running_var += self.momentum * var.data

            return weight * normalized_x + bias
        else:
            broad_mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            broad_var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
            normalized_x = (x - broad_mean) / ops.sqrt(broad_var + self.eps)

            return self.weight.data * normalized_x + self.bias.data


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # NCHW ==> NHWC
        s = x.shape
        _x = x.permute((0, 2, 3, 1)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.permute((0, 3, 1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        reduced_dims = tuple(range(1, len(x.shape)))
        mean = ops.mean(x, reduced_dims, True).broadcast_to(x.shape)
        var = ops.variation(x, reduced_dims, True).broadcast_to(x.shape)
        normalized_x = (x - mean) / ops.sqrt(var + self.eps)

        unsq_param_shape = [1] * len(x.shape)
        unsq_param_shape[-1] = self.dim
        weight = self.weight.reshape(unsq_param_shape).broadcast_to(x.shape)
        bias = self.bias.reshape(unsq_param_shape).broadcast_to(x.shape)

        return weight * normalized_x + bias


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device)
            return mask * x / (1 - self.p)
        else:
            return x


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: Optional[int] = None,
                 bias: int = True, device=None, dtype: str = "float32",
                 flip_kernel: bool = False):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2 if padding is None else padding

        self.flip_kernel = flip_kernel
        self.has_bias = bias

        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels*kernel_size**2,
                out_channels*kernel_size**2,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
                dtype=dtype
            )
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    in_channels*kernel_size**2,
                    1,
                    shape=(out_channels,),
                    gain=3**-0.5,
                    device=device,
                    dtype=dtype
                )
            )
    
    def forward(self, x: Tensor) -> Tensor:
        # NCHW ==> NHWC ==> NH'W'O ==> NOH'W'
        if self.flip_kernel:
            weight = ops.flip(self.weight, (0, 1))
        else:
            weight = self.weight

        _x = x.permute((0, 2, 3, 1))
        outp = ops.conv(_x, weight, self.stride, self.padding)

        if self.has_bias:
            outp += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(outp.shape)
        
        return outp.permute((0, 3, 1, 2))


class ConvTranspose(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0,
                 bias: bool = True, device=None, dtype: str = "float32"):
        super().__init__()

        conv_padding = output_padding + kernel_size - 1 - padding
        self.conv = Conv(in_channels, out_channels, kernel_size, 1,
                         conv_padding, bias, device, dtype, True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        dilated_x = ops.dilate(
            x, axes=(2, 3), dilation=self.stride-1, cut_last=True
        )
        return self.conv(dilated_x)


class MaxPool(Module):
    """
    Multi-channel 2D MaxPool layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=0, stride=kernel_size
    No grouped convolution or dilation
    Only supports square kernels
    Image sizes should be divisible by kernel_size
    """
    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        # NCHW ==> NHWC ==> NH'W'O ==> NOH'W'
        _x = x.permute((0, 2, 3, 1))
        output = ops.maxpool(_x, self.kernel_size)
        return output.permute((0, 3, 1, 2))


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + ops.exp(-x))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        
        bound = 1 / hidden_size**0.5
        self.has_bias = bias

        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    1,
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )

            self.bias_hh = Parameter(
                init.rand(
                    1,
                    hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )
        
        if nonlinearity == 'tanh':
            self.activation = Tanh()
        elif nonlinearity == 'relu':
            self.activation = ReLU()
        else:
            print(f"{nonlinearity} layer is not implemented")
            self.activation = Identity()

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        logits = X @ self.W_ih

        if h is not None:
            logits += h @ self.W_hh

        if self.has_bias:
            total_bias = self.bias_ih + self.bias_hh
            logits += total_bias.broadcast_to(logits.shape)

        return self.activation(logits)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()

        self.n_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]

        for _ in range(num_layers-1):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            )
        
    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(self.n_layers, bs, self.hidden_size, device=X.device,
                requires_grad=True)
            
        h0_arr = [x for x in ops.split(h0, 0)]

        X = ops.split(X, 0)
        output = []

        for i in range(seq_len):
            for j in range(self.n_layers):
                inp = X[i] if j == 0 else h0_arr[j-1]
                h0_arr[j] = self.rnn_cells[j](inp, h0_arr[j])

                if j + 1 == self.n_layers:
                    output.append(h0_arr[j])

        return ops.stack(output, 0), ops.stack(h0_arr, 0)

        
class LSTMCell(Module):
    K_GATES = 4

    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        
        self.has_bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size

        bound = 1 / hidden_size**0.5

        self.W_ih = Parameter(
            init.rand(
                input_size,
                self.K_GATES*hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                self.K_GATES*hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    1,
                    self.K_GATES*hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    1,
                    self.K_GATES*hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype
                )
            )

        self.activations = (Sigmoid(), Sigmoid(), Tanh(), Sigmoid())
        self.tanh = Tanh()

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        logits = X @ self.W_ih
        bs = X.shape[0]

        if h is not None:
            h0, c0 = h
            logits += h0 @ self.W_hh
        else:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype,
                    requires_grad=True)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype,
                    requires_grad=True)

        if self.has_bias:
            total_bias = self.bias_ih + self.bias_hh
            logits += total_bias.broadcast_to(logits.shape)

        gates = ops.split(logits.reshape((bs, self.K_GATES, self.hidden_size)), axis=1)
        gates = [activ(gate) for activ, gate in zip(self.activations, gates)]
        inp_gate, fgt_gate, cell_inp, outp_gate = gates

        new_cell = fgt_gate * c0 + inp_gate * cell_inp
        new_hidden = outp_gate * self.tanh(new_cell)

        return new_hidden, new_cell


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.n_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]

        for _ in range(num_layers-1):
            self.lstm_cells.append(
                LSTMCell(hidden_size, hidden_size, bias, device, dtype)
            )

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        if h is None:
            h0 = init.zeros(self.n_layers, bs, self.hidden_size, device=X.device, 
                requires_grad=True)
            c0 = init.zeros(self.n_layers, bs, self.hidden_size, device=X.device, 
                requires_grad=True)
        else:
            h0, c0 = h
            
        h0_arr = [x for x in ops.split(h0, 0)]
        c0_arr = [x for x in ops.split(c0, 0)]

        X = ops.split(X, 0)
        output = []

        for i in range(seq_len):
            for j in range(self.n_layers):
                inp = X[i] if j == 0 else h0_arr[j-1]
                hid = (h0_arr[j], c0_arr[j])
                h0_arr[j], c0_arr[j] = self.lstm_cells[j](inp, hid)

                if j + 1 == self.n_layers:
                    output.append(h0_arr[j])

        return ops.stack(output, 0), (ops.stack(h0_arr, 0), ops.stack(c0_arr, 0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
