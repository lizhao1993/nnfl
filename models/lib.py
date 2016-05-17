#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-16
Brief:  The library module of neural network
"""

import numpy as np
import logging
import sys
import operator
sys.path.append("../utils/")

logging.basicConfig(
    level=logging.DEBUG,
    format=" [%(levelname)s]%(filename)s:%(lineno)"
    "s[function:%(funcName)s] %(message)s"
)

np.random.seed(1)


def sigmoid(x):
    """
    The sigmoid function. This is numerically-stable version
    x: float
    """
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)


def sigmoid_array(x):
    """
    Numerically-stable sigmoid function
    x: ndarray (float)
    """
    vfunc = np.vectorize(sigmoid)
    return vfunc(x)


class Layer(object):
    """
    Base layer(empty)
    """
    pass


class NormlizationLayer(Layer):
    """
    Normlization layer class
    """
    def __init__(self, n_unit):
        """
        Init layer
        n_unit: int
            Number of unit
        """
        self.n_unit = n_unit
        # To be implementated


class GeneralLayer(Layer):
    """
    General layer class.
    """
    def __init__(self, n_i, n_o, use_bias=True, tfloat='float64'):
        """
        Initialize parameters of softmax layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        self.n_i = n_i
        self.n_o = n_o
        self.use_bias = use_bias
        self.tfloat = tfloat

        # Init weights
        self.w = self.init_weigths()

        self.params = [self.w]
        self.param_names = ['w']
        # Init bias on output layer
        if self.use_bias:
            self.b = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.b)
            self.param_names.append('b')

        # Keep track of lastest forward pass variables
        self.forward_out = None
        self.x = None
        self.gparams = []   # Gradients on parameters

    def init_weigths(self):
        """
        Init weights
        """

        w = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        return w

    def forward(self, x):
        """
        Compute forward pass
        x: numpy.ndarray
            x is the input data with the shape (num_instances, num_inputs)

        output
        --------
        forward_out: numpy.ndarray
            Output with the shape (num_instances, num_inputs)
        """

        if x.shape[1] != self.n_i:
            logging.error("input data shape:%s, not match input unit:%s"
                          % (x.shape, self.n_i))
            raise Exception

        net_input = x.dot(self.w.T)
        if self.use_bias:
            net_input += self.b

        forward_out = self.net_input_to_out(net_input)
        # Keep track it. This will be used in backprop
        self.forward_out = forward_out
        self.x = x
        return forward_out

    def net_input_to_out(self, net_input):
        """
        Net input to out
        net_input: numpy.ndarray
            Net input
        """

        return np.copy(net_input)

    def grad_out_to_net_input(self, go):
        """
        Computing gradients from output to net input
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gradients on the net input
        """

        return np.copy(go)

    def backprop(self, go):
        """
        Back propagation. Note that backprop is only based on the forward pass.
        backprop will choose the lastest forward pass from Multiple forward
        passes.
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gop: numpy.ndarray
            gradients on output of previous layer. The shape of gop is
            (num_instances, num_previus_layer_outputs)
        gparams: self.gparams
        """

        if go.shape[1] != self.n_o:
            logging.error("gradients on output shape:%s, "
                          "not match output unit:%s" % (x.shape, self.n_i))
            raise Exception
        if self.forward_out is None or self.x is None:
            logging.error("No forward computing")
            raise Exception
        if self.x.shape[0] != go.shape[0]:
            logging.error("x shape:%s, gradient shape:%s"
                          % (self.x.shape[0], go.shape))

        # Gradients on net input
        gnet = self.grad_out_to_net_input(go)

        # Gradients on the parameters
        self.gparams = []
        gw = gnet.T.dot(self.x)
        self.gparams.append(gw)
        if self.use_bias:
            gb = gnet.sum(axis=0)
            self.gparams.append(gb)

        # Gradients on output of previous layer
        gop = gnet.dot(self.w)
        return gop


class SoftmaxLayer(GeneralLayer):
    """
    Softmax layer class(Numerically-stable)
    """
    def __init__(self, n_i, n_o, use_bias=True, tfloat='float64'):
        """
        Initialize parameters of softmax layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        GeneralLayer.__init__(self, n_i, n_o, use_bias, tfloat)

    def net_input_to_out(self, net_input):
        """
        Net input to out. Numerically-stable softmax function
        net_input: numpy.ndarray
            Net input
        """

        stable_input = (net_input -
                        np.max(net_input, axis=1)
                        .reshape(net_input.shape[0], 1))
        stable_input_exp = np.exp(stable_input)
        forward_out = (stable_input_exp /
                       np.sum(stable_input_exp, axis=1)
                       .reshape(stable_input_exp.shape[0], 1))

        return forward_out

    def grad_out_to_net_input(self, go):
        """
        Computing gradients from output to net input
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gradients on the net input
        """

        # Gradients on net input
        gnet = np.zeros(go.shape)
        for t in range(0, go.shape[0]):
            go_t = go[t]
            gnet_t = gnet[t]
            tmp_sum = (go_t * self.forward_out[t]).sum()
            for i in range(0, go_t.shape[0]):
                gnet_t[i] = self.forward_out[t][i] * (go_t[i] - tmp_sum)

        return gnet


class HiddenLayer(GeneralLayer):
    """
    Hidden layer class
    """
    def __init__(self, n_i, n_o, act_func='tanh',
                 use_bias=True, tfloat='float64'):
        """
        Initialize parameters of softmax layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        act_func: str
            Activation function. Two values are tanh and sigmoid
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        if act_func not in ['tanh', 'sigmoid']:
            logging.error("act_func:%s, not available")
            raise Exception

        self.act_func = act_func
        GeneralLayer.__init__(self, n_i, n_o, use_bias, tfloat)

    def init_weigths(self):
        """
        Init weights
        """

        w = np.random.uniform(
            low=-np.sqrt(6. / (self.n_i + self.n_o)),
            high=np.sqrt(6. / (self.n_i + self.n_o)),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        if self.act_func == 'sigmoid':
            w *= 4

        return w

    def grad_out_to_net_input(self, go):
        """
        Computing gradients from output to net input
        go: numpy.ndarray
            Gradients on the output of current layer. The shape of go is
            (num_instances, num_outputs)

        output
        --------
        gradients on the net input
        """

        # Gradients on net input
        gnet = np.copy(go)
        if self.act_func == 'tanh':
            gnet = go * (1 - self.forward_out ** 2)
        else:
            gnet = go * self.forward_out * (1 - self.forward_out)

        return gnet

    def net_input_to_out(self, net_input):
        """
        Net input to out. Numerically-stable softmax function
        net_input: numpy.ndarray
            Net input
        """

        if self.act_func == 'tanh':
            forward_out = np.tanh(net_input)
        else:
            forward_out = sigmoid_array(net_input)

        return forward_out


class GradientChecker(object):
    """
    Gradient checker class
    """
    def __init__(self, epsilon=1e-04):
        """
        Init GradientChecker
        epsilon: float
            used during checking
        """

        self.epsilon = epsilon

    def check_layer(self, obj, x, check_params=None):
        """
        Checking gradients of layers
        obj: class instance
            Object to check. obj must provide its params, param_names, forward.
            GradientChecker will use forward function and construct a simple
            loss function (sum function) to test obj's gradients.
        x: numpy.ndarray
            x is the input data of obj
        check_params: list of string
            Parameter names in check_params will be checked. If it is None, all
            parameters will be checked
        """

        if check_params is None:
            check_params = [param_name for param_name in obj.param_names]
        for param_name in check_params:
            param_index = obj.param_names.index(param_name)
            param = obj.params[param_index]
            it = np.nditer(param, flags=['multi_index'],
                           op_flags=['readwrite'])
            gradient_problem = "No problems"
            while not it.finished:
                val_idx = it.multi_index
                val_bak = param[val_idx]
                # Estimated gradients
                param[val_idx] += self.epsilon
                inc_loss = obj.forward(x).sum()
                param[val_idx] = val_bak
                param[val_idx] -= self.epsilon
                dec_loss = obj.forward(x).sum()
                estimated_gradient = (inc_loss - dec_loss) / (2 * self.epsilon)

                # Backprop gradients
                param[val_idx] = val_bak
                obj.forward(x)
                # Gradients on output unit onf obj
                go = np.ones((x.shape[0], obj.n_o), dtype=np.float64)
                obj.backprop(go)
                gparam = obj.gparams[param_index]
                gradient = gparam[val_idx]
                abs_error = abs(gradient - estimated_gradient)
                if (abs_error > 1e-06):
                    # logging.warn("%s gradient problem! error:%f"
                    # % (param_name, abs_error))
                    gradient_problem = "HAVE Problems"
                    break
                it.iternext()
            logging.info("Checking %s, gradient: %s"
                         % (param_name, gradient_problem))

        logging.info("Finish to check gradients")

        return


def layer_test():
    n_i = 5
    n_o = 10
    use_bias = False
    x_num = 20
    x = np.random.uniform(low=0, high=5, size=(x_num, n_i))
    # layer = SoftmaxLayer(n_i, n_o, use_bias)
    layer = HiddenLayer(n_i, n_o, act_func='tanh', use_bias=use_bias)
    # layer = GeneralLayer(n_i, n_o)
    gc = GradientChecker()
    gc.check_layer(layer, x)


if __name__ == "__main__":
    layer_test()
