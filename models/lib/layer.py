#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-16
Brief:  The library of layer
"""

from inc import*
from gradient_checker import GradientChecker


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
        self.forward_out = None
        self.x = None

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
        gnet: numpy.ndarray
            Gradients on net the input of current layer.
        """

        if go.shape[1] != self.n_unit:
            logging.error("shape doesn't match, go shape:%s, unit number:%s"
                          % (go, self.n_unit))
            raise Exception
        if self.forward_out is None or self.x is None:
            logging.error("No forward pass")
            raise Exception

        x_sum = self.x.sum(axis=1).reshape((self.x.shape[0], 1))
        gox_sum = (go * self.x).sum(axis=1).reshape((self.x.shape[0], 1))
        gnet = (go * x_sum - gox_sum) / (x_sum ** 2)

        return gnet

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

        if x.shape[1] != self.n_unit:
            logging.error("input data shape:%s, not match input unit:%s"
                          % (x.shape, self.n_unit))
            raise Exception

        forward_out = x / x.sum(axis=1).reshape((x.shape[0], 1))

        # Keep track of output and input
        self.forward_out = forward_out
        self.x = x

        return forward_out


class GeneralLayer(Layer):
    """
    General layer class.
    """
    def __init__(self):
        # Keep track of lastest forward pass variables
        self.forward_out = None
        self.x = None

    def set_layer(self, w, b=None, use_bias=False):
        """
        Set layer with given params
        w: numpy.ndarry
            The weight of layer which has the shape (n_o, n_i)
        b: numpy.ndarry
            The bias of layer which has the shape (n_o, )
        use_bias: boolean
            Whether to use bias vector on this layer. If b is given and
            use_bias will force to change True. If b is None and use_bias is
            True, the class will init bias
        """
        self.w = w
        self.n_i = w.shape[1]
        self.n_o = w.shape[0]
        self.params = [self.w]
        self.param_names = ['w']
        if b is not None:
            if b.shape[0] != self.n_o:
                logging.error("b is given, but the shape not match w")
                raise Exception
            self.b = b
            self.use_bias = True
        if b is None and use_bias:
            self.use_bias = use_bias
            self.b = np.zeros(shape=self.n_o, dtype=w.dtype)

        if self.use_bias:
            self.params.append(self.b)
            self.param_names.append('b')

    def init_layer(self, n_i, n_o, use_bias=True, tfloat='float64'):
        """
        Initialize parameters of layer.
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

        # Init parameters
        self.init_params()

    def init_params(self):
        """
        Init parameters
        """

        self.w = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)

        self.params = [self.w]
        self.param_names = ['w']
        # Init bias on output layer
        if self.use_bias:
            self.b = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.b)
            self.param_names.append('b')

    def forward(self, x):
        """
        Compute forward pass
        x: numpy.ndarray
            x is the input data with the shape (num_instances, num_inputs)

        output
        --------
        forward_out: numpy.ndarray
            Output with the shape (num_instances, num_outputs)
        """

        if x.shape[1] != self.n_i:
            logging.error("input data shape:%s, not match input unit:%s"
                          % (x.shape, self.n_i))
            raise Exception

        net_input = x.dot(self.w.T)
        if self.use_bias:
            net_input += self.b

        try:
            forward_out = self.net_input_to_out(net_input)
        except:
            logging.error("Failed to compute forward out")
            raise Exception

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
                          "not match output unit:%s" % (go.shape, self.n_o))
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
    def __init__(self):
        GeneralLayer.__init__(self)

    def init_layer(self, n_i, n_o, use_bias=True, tfloat='float64'):
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

        GeneralLayer.init_layer(self, n_i, n_o, use_bias, tfloat)

    def net_input_to_out(self, net_input):
        """
        Net input to out. Numerically-stable softmax function
        net_input: numpy.ndarray
            Net input
        """
        # return GeneralLayer.net_input_to_out(self, net_input)

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
    def __init__(self):
        GeneralLayer.__init__(self)

    def init_layer(self, n_i, n_o, act_func='tanh',
                   use_bias=True, tfloat='float64'):
        """
        Initialize parameters of hidden layer
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
            logging.error("act_func:%s, not available" % act_func)
            raise Exception

        self.act_func = act_func
        GeneralLayer.init_layer(self, n_i, n_o, use_bias, tfloat)

    def init_params(self):
        """
        Init parameters
        """

        self.w = np.random.uniform(
            low=-np.sqrt(6. / (self.n_i + self.n_o)),
            high=np.sqrt(6. / (self.n_i + self.n_o)),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        if self.act_func == 'sigmoid':
            self.w *= 4

        self.params = [self.w]
        self.param_names = ['w']
        # Init bias on output layer
        if self.use_bias:
            self.b = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.b)
            self.param_names.append('b')

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
        Net input to out.
        net_input: numpy.ndarray
            Net input
        """

        if self.act_func == 'tanh':
            forward_out = np.tanh(net_input)
        else:
            forward_out = sigmoid_array(net_input)

        return forward_out


def layer_test():
    n_i = 5
    n_o = 10
    use_bias = True
    x_num = 1
    x = np.random.uniform(low=0, high=5, size=(x_num, n_i))

    softmax_layer = SoftmaxLayer()
    softmax_layer.init_layer(n_i=n_i, n_o=n_o, use_bias=use_bias)
    hidden_layer = HiddenLayer()
    hidden_layer.init_layer(n_i=n_i, n_o=n_o, act_func='sigmoid',
                            use_bias=use_bias)
    norm_layer = NormlizationLayer(x.shape[1])
    general_layer = GeneralLayer()
    general_layer.init_layer(n_i=n_i, n_o=n_o, use_bias=use_bias)
    general_layer_list = [softmax_layer, hidden_layer, general_layer]
    norm_layer_list = [norm_layer]

    gc = GradientChecker()

    for layer in general_layer_list:
        gc.check_layer_params(layer, x)
        gc.check_layer_input(layer, x)
        logging.info("")

    for layer in norm_layer_list:
        gc.check_layer_input(layer, x)


if __name__ == "__main__":
    layer_test()
