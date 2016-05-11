#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/3/24
Brief:  The implementation of Feedward Neural Network (FNN) based on theano
"""

import numpy as np
import timeit
import theano
import theano.tensor as T
import sys
sys.path.append("../utils/")
from tools import*
logging.basicConfig(
    level=logging.DEBUG, 
    format=" [%(levelname)s]%(filename)s:%(lineno)s[function:%(funcName)s] %(message)s"
)

class FNN(object):
    """
    Feedward Neural Network class
    """
    def __init__(self, x, y, word2vec, n_h, n_o, weight=0.5, up_wordvec=False):
        """ Construct the network. All the data types used are declared in ../utils/tools.py

        x: numpy.ndarray, 2d array, train data. Each element in x is word index
        y: numpy.ndarray, 1d array, the correct frame id of train data
        word2vec: numpy.ndarray, 2d array, word vectors
        n_h: int, number of hidden unit
        n_o: int, number of output unit
        weight: float, in [0, 1]. Control the weight between 2-layer NN and 3-layer NN
        up_wordvec: bool, whether to update the word vectors

        """
        if weight > 1. or weight < 0.:
            logging.error("weight not in [0, 1], weight:%f" % weight)
            raise Exception

        # Initializing the parameters of FNN
        # Make shared word2vec
        self.word2vec = theano.shared(word2vec, borrow=True)

        # Init weights between input and hidden layer
        input_dim = x.shape[1] * word2vec.shape[1]
        self.xh = theano.shared(
            np.random.uniform(low=-np.sqrt(6. / (input_dim + n_h)), 
                high=np.sqrt(6. / (input_dim + n_h)),
                size=(input_dim, n_h)).astype(dtype=FLOAT, copy=False), 
            borrow=True
        )
        # Bias on hidden layer
        self.bh = theano.shared(np.zeros(shape=n_h, dtype=FLOAT), borrow=True)

        # Init weights between input and output layer
        self.xo = theano.shared(
            np.random.uniform(low=-np.sqrt(1. / input_dim), 
                high=np.sqrt(1. / input_dim),
                size=(input_dim, n_o)).astype(dtype=FLOAT, copy=False), 
            borrow=True
        )

        # Init weights between hidden and output layer
        self.ho = theano.shared(
            np.random.uniform(low=-np.sqrt(1 / n_h), 
                high=np.sqrt(1 / n_h),
                size=(n_h, n_o)).astype(dtype=FLOAT, copy=False), 
            borrow=True
        )
        # Bias on output layer
        self.bo = theano.shared(np.zeros(shape=n_o, dtype=FLOAT), borrow=True)

        # Parameters of FNN
        self.params = [self.xh, self.xo, self.ho, self.bh, self.bo]
        if up_wordvec:
            self.params.append(self.word2vec)

        # Build symbolic expressions
        x_idxs = T.matrix(dtype=INT) 
        train_y = T.vector(dtype=INT)
        train_x = self.word2vec[x_idxs].reshape(
            shape=(x_idxs.shape[0], input_dim)
        )
        l2_factor = T.scalar(dtype=FLOAT)
        # Learning rate
        lr = T.scalar(dtype=FLOAT)
        hidden_out = T.tanh(T.dot(train_x, self.xh) + self.bh)
        p_y_given_x = T.nnet.softmax(
            (1 - weight) * T.dot(train_x, self.xo) + weight * T.dot(hidden_out, self.ho) + self.bo
        )
        y_preds = T.argmax(p_y_given_x, axis=1)
        zero_one_loss = T.mean(T.neq(y_preds, train_y))
        # Negative log likelihood function
        neg_logll = -T.mean(T.log(p_y_given_x)[T.arange(0, train_y.shape[0]), train_y])
        # L2 norm
        l2_term = (self.xh ** 2).sum() + (self.xo ** 2).sum() + (self.ho ** 2).sum()
        # Cost function
        cost = neg_logll + l2_factor * l2_term
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - lr * gparam) for param, gparam in zip(self.params, gparams)]
        self.train_x = theano.function(
            inputs=[x_idxs, train_y, lr, l2_factor],
            outputs=cost,
            updates=updates
        )
        self.zero_one_loss = theano.function(inputs=[x_idxs, train_y], outputs=zero_one_loss)
        self.predict_x = theano.function(inputs=[x_idxs], outputs=y_preds)
        self.x = x
        self.y = y
    
    def __train_x_with_minibatch(self, minibatch, lr, l2_factor=0.001):
        """
        Training FNN with one eopch with minibatch. This method is private
        minibatch: int
        lr: float, learning rate
        l2_factor: float, the factor of L2 norm

        """
        n_batches = int(len(self.y) / minibatch)
        for batch_i in range(0, n_batches):
            self.train_x(
                self.x[batch_i * minibatch:(batch_i + 1) * minibatch],
                self.y[batch_i * minibatch:(batch_i + 1) * minibatch],
                lr, l2_factor
            )
        if n_batches * minibatch != len(self.y):
            # Train the rest 
            self.train_x(
                self.x[(batch_i + 1) * minibatch:],
                self.y[(batch_i + 1) * minibatch:],
                lr, l2_factor
            )

    def train(self, minibatch, lr, max_epochs=100, l2_factor=0.0001, verbose=False):
        """
        Training FNN. Training will be stopped when the zero-one loss is zero on train data

        max_epochs: int, max epoch of training
        minibatch: int
        lr: float, learning rate
        l2_factor: float, the factor of L2 norm
        verbose: bool, whether to print information during each epoch training
        return [the epoch, running time(m)]
        """

        start_time = timeit.default_timer()
        for epoch in range(1, max_epochs + 1):
            self.__train_x_with_minibatch(minibatch, lr, l2_factor)
            error = self.zero_one_loss(self.x, self.y)
            if verbose:
                print("epoch: %d training, zero-one loss on train data: %f" % (epoch, error))
            if abs(error - 0.0) <= 0.00001:
                break
        end_time = timeit.default_timer()
        return [epoch, (end_time - start_time) / 60.]

    def early_stopping_train(self, validation, minibatch, lr, \
            l2_factor=0.0001, max_epochs=100, patience=15,\
            improvement_threould=0.995, validation_freq=3, verbose=False):
        """
        Training FNN. Training will be encouraged with significant improvement on validation set
            and will be penalized with slightly or negative improvement on validation set

        validation: [validation_x, validation_y], each emlement of this list is a numpy.ndarray
        minibatch: int
        lr: float, learning rate
        l2_factor: float, the factor of L2 norm
        max_epochs: int, the max epochs during the training
        patience: int, the training will be stopped when the epoch is bigger than patience
        improvement_threshold: float, zero-one loss reduction factor
        verbose: bool, whether to print information during each epoch training
        return [the epoch, running time(m)]
        """
    
        patience_increase = validation_freq
        best_validation_loss = np.inf
        start_time = timeit.default_timer()
        for epoch in range(1, max_epochs + 1):
            self.__train_x_with_minibatch(minibatch, lr, l2_factor)
            if epoch % validation_freq == 0:
                current_validation_loss = self.zero_one_loss(validation[0], validation[1])
                if current_validation_loss < best_validation_loss:
                    # Significant improvement
                    if current_validation_loss < best_validation_loss * improvement_threould:
                        patience = max(patience, epoch + patience_increase)
                    else: # Slightly improvement
                        patience -= validation_freq
                    best_validation_loss = current_validation_loss
                else: # negative improvement
                    patience -= validation_freq
                current_train_loss = self.zero_one_loss(self.x, self.y)
                if verbose:
                    print(
                        "epoch:%d training\tpatience:%d\tbest "\
                                "zero-one loss on validation:%f\tcurrent zero-loss on train:%f" \
                            % (epoch, patience, best_validation_loss, current_train_loss)
                    )
                # Stop training when loss is zero on train data.
                #       This avoid patience value is too large
                if abs(current_train_loss - 0.0) <= 0.00001:
                    break
            # Stop training without patience
            if epoch > patience:
                break

        end_time = timeit.default_timer()
        return [epoch, (end_time - start_time) / 60.]

    def predict(self, x):
        """
        FNN predict
        x: numpy.ndarray
        return numpy.ndarray, the predict result
        """
        return self.predict_x(x)


def fnn_test():
    x_row = 100
    x_col = 20
    n_o = 10
    n_h = 30
    vocab_size = 200
    word_dim = 12
    weight = 0.5
    up_wordvec = True

    x = np.random.randint(
        low=0, high=vocab_size, size=(x_row, x_col)
    ).astype(dtype=INT, copy=False)
    y = np.random.randint(low=0, high=n_o, size=x_row).astype(dtype=INT, copy=False)
    word2vec = np.random.uniform(
        low=0, high=2, size=(vocab_size, word_dim)
    ).astype(dtype=FLOAT, copy=False)
    fnn = FNN(x, y, word2vec, n_h, n_o, weight, up_wordvec)

    max_epochs = 100
    minibatch = 10
    lr = 0.1
    l2_factor = 0.0001
    # Training 
    # epoch, running_time = fnn.train(minibatch, lr, max_epochs, l2_factor, verbose=True)
    # print("epoch: %d\trunning_time:%fm" % (epoch, running_time))

    # Training with early-stopping
    validation_x = x[0:int(x_row / 4)]
    validation_y = y[0:int(x_row / 4)]
    validation = [validation_x, validation_y]
    epoch, running_time = fnn.early_stopping_train(validation, minibatch, lr, \
        l2_factor=0.0001, max_epochs=100, patience=15,\
        improvement_threould=0.995, validation_freq=3, verbose=True)
    print("epoch: %d\trunning_time:%fm" % (epoch, running_time))


if __name__ == "__main__":
    fnn_test()


