#! /usr/bin/env python3
"""
Authors: fengyukun
Date:  2016-05-26
Brief:  The implementation of long short-term memory (LSTM) layer
"""

from inc import*
from gradient_checker import GradientChecker
from layer import Layer


class LSTMLayer(Layer):
    """
    The long short-term memory (LSTM) layer class
    """
    def __init__(self):
        pass

    def init_layer(self, n_i, n_o, act_func='tanh',
                   use_bias=True, tfloat='float64'):
        """
        Initialize parameters of LSTM layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer. Here it it the number of
            blocks
        act_func: str
            Activation function used for cell input and output.
            Two values are tanh and sigmoid
        use_bias: boolean
            Whether to use bias vector on this layer
        tfloat: str
            Type of float used on the weights
        """

        self.n_i = n_i
        self.n_o = n_o
        self.act_func = act_func
        self.use_bias = use_bias
        self.tfloat = tfloat
        self.init_params()

    def init_params(self):
        """
        Init parameters
        """

        self.params = []
        self.param_names = []

        # Weigths between input x and input gates
        self.wxi = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxi)
        self.param_names.append("wxi")
        if self.use_bias:
            self.ib = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.ib)
            self.param_names.append("ib")

        # Weights between input x and forget gates
        self.wxf = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxf)
        self.param_names.append("wxf")
        if self.use_bias:
            self.fb = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.fb)
            self.param_names.append("fb")

        # Weigths between input x and cell
        self.wxc = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxc)
        self.param_names.append("wxc")
        if self.use_bias:
            self.cb = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.cb)
            self.param_names.append("cb")

        # Weigths between input x and output gates
        self.wxo = np.random.uniform(
            low=-np.sqrt(1. / self.n_i),
            high=np.sqrt(1. / self.n_i),
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.wxo)
        self.param_names.append("wxo")
        if self.use_bias:
            self.ob = np.zeros(shape=self.n_o, dtype=self.tfloat)
            self.params.append(self.ob)
            self.param_names.append("ob")

        # Recurrent weights init

        # Weights between previous hidden output and input gates
        self.whi = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.whi)
        self.param_names.append("whi")

        # Weights between previous hidden output and cell
        self.whc = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.whc)
        self.param_names.append("whc")

        # Weights between previous hidden output and forget gates
        self.whf = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.whf)
        self.param_names.append("whf")

        # Weights between previous hidden output and output gates
        self.who = np.random.uniform(
            low=-np.sqrt(1. / self.n_o),
            high=np.sqrt(1. / self.n_o),
            size=(self.n_o, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)
        self.params.append(self.who)
        self.param_names.append("who")

    def set_layer(self):
        """
        Set layer with given params
        """
        pass

    def single_forward(self, x_t, ht_1, ct_1):
        """
        Computing forward in a single pass at time t.
        x_t: numpy.ndarray
            The input data whose shape is (self.n_i, )
        ht_1: numpy.ndarray
            The output of hidden at t - 1. The shape is (self.n_o, )
        ct_1: numpy.ndarray
            The output of cell at t - 1. The shape is (self.n_o, )

        Return
        ---------
        ct: numpy.ndarray
            The output of cell at time t
        ot: numpy.ndarray
            The output of output gates at time t
        scaled_oct: numpy.ndarray
            Scaled output of cell at time t
        scaled_incellt: numpy.ndarray
            Scaled input of cell at time t
        it: numpy.ndarray
            The output of input gates at time t
        xt: numpy.ndarray
            The input of blocks at time t
        ht: numpy.ndarray
            The output of blocks at time t
        ft: numpy.ndarray
            The output of forget gates at time t
        """

        # Input gate at time t
        it = self.wxi.dot(x_t) + self.whi.dot(ht_1)
        if self.use_bias:
            it += self.ib
        it = sigmoid_array(it)

        # Forget gate at time t
        ft = self.wxf.dot(x_t) + self.whf.dot(ht_1)
        if self.use_bias:
            ft += self.fb
        ft = sigmoid_array(ft)

        # Cell output at time t
        scaled_incellt = self.wxc.dot(x_t) + self.whc.dot(ht_1)
        if self.use_bias:
            scaled_incellt += self.cb
        if self.act_func == 'tanh':
            scaled_incellt = np.tanh(scaled_incellt)
        else:
            scaled_incellt = sigmoid_array(scaled_incellt)
        ct = it * scaled_incellt + ct_1 * ft

        # Output gate
        ot = self.wxo.dot(x_t) + self.who.dot(ht_1)
        if self.use_bias:
            ot += self.ob
        ot = sigmoid_array(ot)

        if self.act_func == 'tanh':
            scaled_oct = np.tanh(ct)
        else:
            scaled_oct = sigmoid_array(ct)
        ht = ot * scaled_oct

        return (ct, ot, scaled_oct, scaled_incellt, it, ht, ft)

    def single_backprop(self, ght, gct, ct, ot, ct_1, scaled_oct,
                        scaled_incellt, it, xt, ht_1, ft):
        """
        Backprop in a single pass at time t.
        ght: numpy.ndarray
            Accumulated gradients on output of blocks at time t
        gct: numpy.ndarray
            Accumulated gradients on output of cell at time t
        ct: numpy.ndarray
            The output of cell at time t
        ot: numpy.ndarray
            The output of output gates at time t
        ct_1: numpy.ndarray
            The output of cell at time t - 1
        scaled_oct: numpy.ndarray
            Scaled output of cell at time t
        scaled_incellt: numpy.ndarray
            Scaled input of cell at time t
        it: numpy.ndarray
            The output of input gates at time t
        xt: numpy.ndarray
            The input of blocks at time t
        ht_1: numpy.ndarray
            The output of blocks at time t - 1
        ft: numpy.ndarray
            The output of forget gates at time t

        Returns
        ------------
        gxt: numpy.ndarray
            Gradients on xt
        ght_1: numpy.ndarray
            Gradients on output of blocks at t - 1
        gct_1: numpy.ndarray
            Gradients on output of cell at t - 1
        """

        # Gradients on output of cell from top of blocks
        gcell = ght * ot
        if self.act_func == 'tanh':
            gcell *= (1 - scaled_oct ** 2)
        else:
            gcell *= scaled_oct * (1 - scaled_oct)
        # Accumulated gradients on cell
        gcell += gct
        # Gradients on cell at t - 1
        gct_1 = gcell * ft

        # Gradients on output gates
        gogates = ght * scaled_oct
        # Gradients on forget gates
        gfgates = gcell * ct_1
        # Gradients on input gates
        gigates = gcell * scaled_incellt
        # Gradients on input of cell
        gincell = gcell * it
        if self.act_func == 'tanh':
            gincell *= (1 - scaled_incellt ** 2)
        else:
            gincell *= scaled_incellt * (1 - scaled_incellt)

        # Gradients on input of output gates
        giogates = gogates * ot * (1 - ot)
        # Gradients on wxo, who and ob
        self.gwxo += giogates.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwho += giogates.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gob += giogates
        # Gradients on xt
        gxt = 0
        gxt += self.wxo.T.dot(giogates)
        # Gradients on previous output of blocks
        ght_1 = 0
        ght_1 += self.who.T.dot(giogates)

        # Gradients on input of forget gates
        gifgates = gfgates * ft * (1 - ft)
        # Gradients on wxf, whf and fb
        self.gwxf += gifgates.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwhf += gifgates.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gfb += gifgates
        # Gradients on xt
        gxt += self.wxf.T.dot(gifgates)
        # Gradients on previous output of blocks
        ght_1 += self.whf.T.dot(gifgates)

        # Gradients on input of input gates
        giigates = gigates * it * (1 - it)
        # Gradients on wxi, whi and ib
        self.gwxi += giigates.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwhi += giigates.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gib += giigates
        # Gradients on xt
        gxt += self.wxi.T.dot(giigates)
        ght_1 += self.whi.T.dot(giigates)

        # Gradients on input of cell
        self.gwxc += gincell.reshape((self.n_o, 1)).dot(
            xt.reshape((1, self.n_i))
        )
        self.gwhc += gincell.reshape((self.n_o, 1)).dot(
            ht_1.reshape((1, self.n_o))
        )
        if self.use_bias:
            self.gcb += gincell
        # Gradients on xt
        gxt += self.wxc.T.dot(gincell)
        ght_1 += self.whc.T.dot(gincell)

        return (gxt, ght_1, gct_1)

    def forward(self, x):
        """
        x: 3d array-like, In the whole it usually is jagged array. The first
        loop is sample numbers. The second is unit representation numbers. The
        third is float numbers in one unit
        --------
        """

        self.cts = []
        self.ots = []
        self.scaled_octs = []
        self.scaled_incellts = []
        self.its = []
        self.hts = []
        self.fts = []

        # Iterate each sample over x
        for i in range(0, len(x)):
            cts = []
            ots = []
            scaled_octs = []
            scaled_incellts = []
            its = []
            hts = []
            fts = []
            # Iterate each unit over x[i]
            ht_1 = np.zeros((self.n_o, ))
            ct_1 = np.zeros((self.n_o, ))
            for t in range(0, len(x[i])):
                (ct, ot, scaled_oct, scaled_incellt, it, ht, ft) = (
                    self.single_forward(np.asarray(x[i][t]), ht_1, ct_1)
                )
                ht_1 = ht
                ct_1 = ct
                cts.append(ct)
                ots.append(ot)
                scaled_octs.append(scaled_oct)
                scaled_incellts.append(scaled_incellt)
                its.append(it)
                hts.append(ht)
                fts.append(ft)

            # Keep track middle varibles
            self.x = x
            self.cts.append(cts)
            self.ots.append(ots)
            self.scaled_octs.append(scaled_octs)
            self.scaled_incellts.append(scaled_incellts)
            self.its.append(its)
            self.hts.append(hts)
            self.fts.append(fts)

        return self.hts

    def backprop(self, go):
        """
        Back propagation. Note that backprop is only based on the forward pass.
        backprop will choose the lastest forward pass from Multiple forward
        passes.
        go: 3d array-like
            Gradients on the output of current layer.

        output
        --------
        gop: numpy.ndarray
            gradients on output of previous layer. The shape of gop is the
            same as x
        gparams: self.gparams
        """

        if not hasattr(self, 'x'):
            logging.error("No forward pass is computed")
            raise Exception

        gop = np.copy(self.x)
        # Init gradients on parameters
        self.gparams = []
        self.gwxi = np.zeros(self.wxi.shape)
        self.gparams.append(self.gwxi)
        if self.use_bias:
            self.gib = np.zeros(self.ib.shape)
            self.gparams.append(self.gib)

        self.gwxf = np.zeros(self.wxf.shape)
        self.gparams.append(self.gwxf)
        if self.use_bias:
            self.gfb = np.zeros(self.fb.shape)
            self.gparams.append(self.gfb)

        self.gwxc = np.zeros(self.wxc.shape)
        self.gparams.append(self.gwxc)
        if self.use_bias:
            self.gcb = np.zeros(self.cb.shape)
            self.gparams.append(self.gcb)

        self.gwxo = np.zeros(self.wxo.shape)
        self.gparams.append(self.gwxo)
        if self.use_bias:
            self.gob = np.zeros(self.ob.shape)
            self.gparams.append(self.gob)

        # Gradients on recurrent weights
        self.gwhi = np.zeros(self.whi.shape)
        self.gparams.append(self.gwhi)
        self.gwhc = np.zeros(self.whc.shape)
        self.gparams.append(self.gwhc)
        self.gwhf = np.zeros(self.whf.shape)
        self.gparams.append(self.gwhf)
        self.gwho = np.zeros(self.who.shape)
        self.gparams.append(self.gwho)

        for i in range(0, len(go)):
            ght = np.zeros((self.n_o, ))
            gct = np.zeros((self.n_o, ))
            for t in range(len(go[i]) - 1, -1, -1):
                if go[i][t] is not None:
                    ght += go[i][t]
                if t == 0:
                    ht_1 = np.zeros((self.n_o, ))
                    ct_1 = np.zeros((self.n_o, ))
                else:
                    ht_1 = self.hts[i][t - 1]
                    ct_1 = self.cts[i][t - 1]
                (gxt, ght_1, gct_1) = self.single_backprop(
                    ght, gct, self.cts[i][t], self.ots[i][t],
                    ct_1, self.scaled_octs[i][t],
                    self.scaled_incellts[i][t], self.its[i][t], self.x[i][t],
                    ht_1, self.fts[i][t]
                )

                gop[i][t] = gxt
                ght = ght_1
                gct = gct_1

        return gop


def layer_test():
    n_i = 3
    n_o = 5
    use_bias = True
    x_num = 3
    x = []
    go = []
    # Construct x and go
    for i in range(0, x_num):
        # Random column
        col = np.random.randint(low=1, high=4)
        x_row = []
        go_row = []
        for j in range(0, col):
            x_row.append(np.random.uniform(low=0, high=5, size=(n_i, )))
            go_row.append(np.random.uniform(low=0, high=5, size=(n_o, )))
        x.append(np.asarray(x_row))
        go.append(np.asarray(go_row))

    lstm_layer = LSTMLayer()
    lstm_layer.init_layer(n_i=n_i, n_o=n_o, act_func='sigmoid',
                          use_bias=use_bias)

    gc = GradientChecker(epsilon=1e-05)
    gc.check_jagged_input(lstm_layer, x)
    check_params = None
    gc.check_layer_params(lstm_layer, x, check_params)

if __name__ == "__main__":
    layer_test()
