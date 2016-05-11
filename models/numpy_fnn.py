#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/4/20
Brief:  The implementation of Feedward Neural Network (FNN) based on 
        ptre-trained word vectors
"""

import numpy as np
import logging
import sys
import operator
sys.path.append("../utils/")
import metrics

logging.basicConfig(
    level=logging.DEBUG, 
    format=" [%(levelname)s]%(filename)s:%(lineno)"\
        "s[function:%(funcName)s] %(message)s"
)

np.random.seed(1)

class FNN(object):
    """ Feedward Neural Network class with the direct connection from input to 
        output
    """
    def __init__(self, x, label_y, n_h, word2vec, lambda1=0.5, \
                lambda2=0.5, beta=1e-04, up_wordvec=False, tfloat='float64'):
        """ Initialize the network. The network has a direct 
            connection from input to output. The activition function will 
            be hyperbolic tangent function in hidden layer

        x: numpy.ndarray, 2d array, train data. Each element in x is word 
            index(int). Word index should start from 0 and correspond with row 
            numbers of word2vec. One row of  x represents a sentence
        label_y: numpy.ndarray, 1d array, the correct label of train data

        n_h: int, number of hidden unit
        word2vec: numpy.ndarray, 2d array, word vectors. Each row represents 
                word vectors. E.g., word_vectors = word2vec[word_index] 
        lambda1, lambda2: float, lambda1 controls the weight of  
                3-layer FNN. lambda2 controls the weight of direct connection 
                from input to output layer. The mathematical expression can be
                expressed as:
                net input of softmax of FNN = lambda1 * output of hidden layer
                    + lambda2 * (input * the weight matrix from input to output)
        beta: float, l2 regularization parameter
        up_wordvec: bool, whether to update the word vectors
        tfloat: string, the type of float for weight in FNN

        """

        self.x = x
        self.n_h = n_h
        self.word2vec = word2vec
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.beta = beta
        self.tfloat = tfloat
        self.up_wordvec = up_wordvec
        # Number of input unit
        self.n_i = self.x.shape[1] * self.word2vec.shape[1]

        # label_y should be normalized to continuous integers starting from 0
        self.label_y = label_y
        label_set = set(self.label_y)
        y_set = np.arange(0, len(label_set))
        label_to_y = dict(zip(label_set, y_set))
        self.y = np.array([label_to_y[label] for label in self.label_y])

        # Record the map from label id to label for furthur output
        self.y_to_label = {k : v for v, k in label_to_y.items()}
        self.n_o = y_set.shape[0]   # Number of output unit

        # Init weights between input and hidden layer (note FNN use tanh in the
        # hidden layer)
        self.wih = np.random.uniform(
            low=-np.sqrt(6. / (self.n_i + self.n_h)), 
            high=np.sqrt(6. / (self.n_i + self.n_h)),
            size=(self.n_i, self.n_h)
        ).astype(dtype=self.tfloat, copy=False)

        # Init bias on hidden layer
        self.bh = np.zeros(shape=self.n_h, dtype=self.tfloat)

        # Init weights between hidden and output layer
        self.who = np.random.uniform(
            low=-np.sqrt(1. / self.n_h), 
            high=np.sqrt(1. / self.n_h), 
            size=(self.n_h, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)

        # Init bias on output layer
        self.bo = np.zeros(shape=self.n_o, dtype=self.tfloat)

        # Init weights between input and output layer (direct connection)
        self.wio = np.random.uniform(
            low=-np.sqrt(1. / self.n_i), 
            high=np.sqrt(1. / self.n_i), 
            size=(self.n_i, self.n_o)
        ).astype(dtype=self.tfloat, copy=False)

        # Parameters of FNN
        self.params = [self.who, self.bo, self.wih, self.bh, self.wio]
        self.params_names = ['who', 'bo', 'wih', 'bh', 'wio']

    def __forward_propagation(self, x):
        """
        Forward propagation of FNN

        x: numpy.ndarray, 2d array, train data. Each element in x is word index 

        return [py, hout, vectorized_x]
            py: numpy.ndarray, the output of FNN
            hout: numpy.ndarray, the output of hidden layer used for 
                backpropagation. This will save the time to recompute hout 
                in backpropagation
            vectorized_x: numpy.ndarray, 2d array. vectorized_x is vectorized 
                data by word2vec. This will be used by backpropagation
        """

        # Check the input unit
        if self.n_i != x.shape[1] * self.word2vec.shape[1]:
            logging.error("The dimension of train data and the number "\
                "of input units are different")
            raise Exception

        vectorized_x = self.word2vec[x].reshape(
            (x.shape[0], self.n_i)
        )

        # hidden net input
        hnet = np.dot(vectorized_x, self.wih) + self.bh
        # hidden out
        hout = np.tanh(hnet)
        # Output net
        onet = self.lambda1 * (np.dot(hout, self.who) + self.bo) \
                    + self.lambda2 * np.dot(vectorized_x, self.wio)
        # Softmax function for classification
        safe_onet = onet - np.max(onet, axis=1).reshape(onet.shape[0], 1)
        safe_exp_onet = np.exp(safe_onet)
        py = safe_exp_onet /\
                np.sum(safe_exp_onet, axis=1).reshape(onet.shape[0], 1)
            
        return [py, hout, vectorized_x]

    def predict(self, x):
        """
        Prediction of FNN on x

        x: numpy.ndarray, 2d array, train data. Each element in x is word 
            index(int). Word index should start from 0 and correspond with row 
            numbers of word2vec. One row of  x represents a sentence
        return: numpy.ndarray, 1d array. The predict label on x
        """
        try:
            py, _, _ = self.__forward_propagation(x)
        except:
            logging.error("Failed to run forward propagation")
            raise Exception

        y = py.argmax(axis=1)
        return np.array([self.y_to_label[i] for i in y])

    def minibatch_train(self, lr=0.1, minibatch=5, \
                         max_epochs=100, verbose=False):
        """
        Minibatch training. Training will be stopped when the 
        zero-one loss is zero on x

        lr: float, learning rate
        minibatch: int
        max_epochs: the max epoch, int
        verbose: bool, whether to print information during each epoch training
        return: int, epoch
        """

        for epoch in range(1, max_epochs + 1):
            n_batches = int(self.y.shape[0] / minibatch)
            for batch_i in range(0, n_batches):
                self.__one_batch_train(
                    self.x[batch_i * minibatch:(batch_i + 1) * minibatch],
                    self.y[batch_i * minibatch:(batch_i + 1) * minibatch],
                    lr 
                )
            # Train the rest if it has
            if n_batches * minibatch != self.y.shape[0]:
                self.__one_batch_train(
                    self.x[(batch_i + 1) * minibatch:],
                    self.y[(batch_i + 1) * minibatch:],
                    lr
                )
            label_preds = self.predict(self.x)
            error = metrics.zero_one_loss(self.label_y, label_preds)
            loss = self.calculate_total_loss()
            if verbose:
                logging.info("epoch: %d training,on train data, " \
                    "cross-entropy:%f, zero-one loss: %f" \
                        % (epoch, loss, error))
            if abs(error - 0.0) <= 0.00001:
                break
        return epoch

    def calculate_total_loss(self):
        """
        Calculate the loss on x given the right label y

        return total loss (cross entropy plus l2 regularization)

        """

        try:
            py, _, _ = self.__forward_propagation(self.x)
        except:
            logging.error("Failed to run forward propagation")
            raise Exception

        cross_entropy = -np.sum(
            np.log(py[np.arange(0, self.y.shape[0]), self.y])
        )
        l2_term = (self.wih ** 2).sum() + (self.who ** 2).sum() \
                    + (self.wio ** 2).sum()
        return cross_entropy + self.beta * l2_term

    def calculate_gradients(self, x, y):
        """
        Calculate gradidents on parameters of FNN

        x: numpy.ndarray, 2d array, train data. Each element in x is word 
            index(int). Word index should start from 0 and correspond with row 
            numbers of word2vec. One row of  x represents a sentence
        y: numpy.ndarray, 1d array, the normalized label(int) of train data

        if not update word vectors
        return [gwho, gbo, gwih, gbh, gwio, None, None]
        if update word vectors
        return [gwho, gbo, gwih, gbh, gwio, wvidxs, gvectors]
        
        """

        try:
            py, hout, vectorized_x = self.__forward_propagation(x)
        except:
            logging.error("Failed to run forward propagation")
            raise Exception

        # Backpropagation with batch gradient descent (bgd)

        # Gradient of loss on the net input of output layer.
        gonet = py
        gonet[np.arange(0, len(gonet)), y] -= 1
        # Gradident on the output of hidden layer
        ghout = np.dot(self.lambda1 * gonet, self.who.T)
        # Gradident on the net input of hidden layer
        ghnet = ghout * (1 - hout ** 2)

        # Gradients on parameters
        gwho = np.dot(hout.T, self.lambda1 * gonet) + self.who * 2 * self.beta
        gbo = (self.lambda1 * gonet).sum(axis=0)
        gwih = np.dot(vectorized_x.T, ghnet) + self.wih * 2 * self.beta
        gbh = ghnet.sum(axis=0)
        gwio = np.dot(vectorized_x.T, self.lambda2 * gonet) \
                + self.wio * 2 * self.beta
        # The order is same as self.params
        gparams = [gwho, gbo, gwih, gbh, gwio, None, None]
        if self.up_wordvec:
            gx = np.dot(ghnet, self.wih.T) \
                + np.dot(self.lambda2 * gonet, self.wio.T)
            wvidxs = []     # word vectors index 
            gvectors = []   # gradients on vectors
            for  sample, gsample in zip(x, gx):
                for i in range(0, len(sample)):
                    wvidx = sample[i]   # word vectors index
                    dw = self.word2vec.shape[1] # dimension of word vectors
                    gvector = gsample[i * dw:(i + 1) * dw]
                    # Accumulate gradients on the same vector
                    if wvidx in wvidxs:
                        gvectors[wvidxs.index(wvidx)] += gvector
                    else:
                        wvidxs.append(wvidx)
                        gvectors.append(gvector)
            gparams[-2] = wvidxs
            gparams[-1] = gvectors
        return gparams

    def gradient_check(self):
        """
        Gradient checking.
        """

        epsilon = 1e-04
        check_params = ['who', 'bo', 'wih', 'bh', 'wio']
        for param_name in check_params:
            param_index = self.params_names.index(param_name)
            param = self.params[param_index]
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            gradient_problem = False
            while not it.finished:
                val_idx = it.multi_index
                val_bak = param[val_idx]
                # Estimated gradients
                param[val_idx] += epsilon
                inc_loss = self.calculate_total_loss()
                param[val_idx] = val_bak
                param[val_idx] -= epsilon
                dec_loss = self.calculate_total_loss()
                estimated_gradient = (inc_loss - dec_loss) / (2 * epsilon)

                # Backprop gradients
                param[val_idx] = val_bak
                gparam = self.calculate_gradients(self.x, self.y)[param_index]
                gradient = gparam[val_idx]
                abs_error = abs(gradient - estimated_gradient)
                if (abs_error > 1e-06):
                    logging.warn("%s gradient problem!" % param_name) 
                    gradient_problem = True
                    break
                it.iternext()
            logging.info("Checking %s, gradient problem: %s" \
                        % (param_name, gradient_problem))

        # Checking gradients on word vectors
        if self.up_wordvec:
            gradient_problem = False
            for word_index in np.unique(self.x):
                word_vector = self.word2vec[word_index]
                for i in range(0, word_vector.shape[0]):
                    bak = word_vector[i]
                    # Estimated gradients
                    word_vector[i] += epsilon
                    inc_loss = self.calculate_total_loss()
                    word_vector[i] = bak
                    word_vector[i] -= epsilon
                    dec_loss = self.calculate_total_loss()
                    estimated_gradient = (inc_loss - dec_loss) / (2 * epsilon)
                    # Backprop gradients
                    word_vector[i] = bak
                    wvidxs, gvectors = self.calculate_gradients(
                        self.x, self.y
                    )[-2:]
                    gradient = gvectors[wvidxs.index(word_index)][i]
                    abs_error = abs(gradient - estimated_gradient)
                    if (abs_error > 1e-06):
                        gradient_problem = True
                        break
                if gradient_problem:
                    break
            logging.info("Checking word vectors, gradient problem:%s" \
                            % gradient_problem)

        logging.info("Finish to check gradients")

        return

    def __one_batch_train(self, x, y, lr=0.1):
        """
        One epoch batch training of FNN on x given right label y

        x: numpy.ndarray, 2d array, train data. Each element in x is word 
            index(int). Word index should start from 0 and correspond with row 
            numbers of word2vec. One row of  x represents a sentence
        y: numpy.ndarray, 1d array, the normalized label(int) of train data

        lr: float, learning rate
        """

        # Calculate the gradients on parameters of FNN
        try:
            gparams = self.calculate_gradients(x, y) 
        except:
            logging.error("Failed to calculate gradients")
            raise Exception

        # Update parameters
        for gparam, param in zip(gparams[0:len(self.params)], self.params):
            param -= lr * gparam
        if self.up_wordvec:
            for wvidx, gvector in zip(gparams[-2], gparams[-1]):
                self.word2vec[wvidx] -= lr * gvector
        return


def fnn_test():
    n_h = 30
    train_row = 100
    train_col = 5
    vocab_size = 200
    word_dim = 2 

    word2vec = np.random.uniform(low=0, high=3, size=(vocab_size, word_dim))
    x = np.random.randint(
        low=0, high=len(word2vec), size=(train_row, train_col)
    )
    label_y = np.random.randint(low=3, high=13, size=len(x))

    # Training setting
    up_wordvec = True
    lr = 1e-02
    minibatch = train_row
    verbose = True
    lambda1 = 1
    lambda2 = 1

    fnn = FNN(x, label_y, n_h=n_h, lambda1=lambda1, lambda2=lambda2, 
                up_wordvec=up_wordvec, word2vec=word2vec)

    # fnn.gradient_check()
    # return
    fnn.minibatch_train(
        lr=lr, minibatch=minibatch, verbose=verbose
    )

def simple_fnn_test():
    n_h = 30
    vocab_size = 200
    word_dim = 2 

    x = np.array(
        [[1, 2, 1], 
         [2, 1, 2], 
         [2, 1, 1], 
         [3, 8, 8], 
         [8, 8, 3], 
        ]
    )
    label_y = np.array([0, 1, 0, 2, 2])
    word2vec = np.random.uniform(low=0, high=3, size=(vocab_size, word_dim))

    test_x = np.array(
    [[1, 1, 2], 
     [2, 2, 2], 
     [1, 1, 3], 
     [8, 8, 4]]
    )
    y_true = np.array([0, 1, 0, 2])

    # Training setting
    up_wordvec = True
    lr = 1e-01
    minibatch = x.shape[0]
    verbose = True
    lambda1 = 0
    lambda2 = 1

    fnn = FNN(x, label_y, n_h=n_h, lambda1=lambda1, lambda2=lambda2, 
                up_wordvec=up_wordvec, word2vec=word2vec)

    # fnn.gradient_check()
    # return
    fnn.minibatch_train(
        lr=lr, minibatch=minibatch, verbose=verbose
    )
    # Test
    y_pred = fnn.predict(test_x)
    print(y_pred)
    print(metrics.zero_one_loss(y_true, y_pred))

if __name__ == "__main__":
    # fnn_test()
    simple_fnn_test()


