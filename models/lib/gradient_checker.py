#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-16
Brief:  The library module of neural network
"""

from inc import*


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

    def check_layer_net_input(self, obj, x):
        """
        Checking gradients on net input of given layer
        obj: Layer
            Object to check. obj must provide its forward, backprop.
            GradientChecker will use forward function and construct a
            simple loss function (sum function) to test obj's gradients.
        """

        logging.info("To check the gradients on net input")

        param = np.copy(x)
        it = np.nditer(param, flags=['multi_index'],
                       op_flags=['readwrite'])
        gradient_problem = "No problems"
        while not it.finished:
            val_idx = it.multi_index
            val_bak = param[val_idx]
            # Estimated gradients
            param[val_idx] += self.epsilon
            inc_loss = obj.forward(param).sum()
            param[val_idx] = val_bak
            param[val_idx] -= self.epsilon
            dec_loss = obj.forward(param).sum()
            estimated_gradient = (inc_loss - dec_loss) / (2 * self.epsilon)

            # Backprop gradients
            param[val_idx] = val_bak
            obj.forward(param)
            # Gradients on output unit onf obj
            go = np.ones(param.shape, dtype=np.float64)
            gparam = obj.backprop(go)
            gradient = gparam[val_idx]
            abs_error = abs(gradient - estimated_gradient)
            if (abs_error > 1e-06):
                # logging.warn("%s gradient problem! error:%f"
                # % (param_name, abs_error))
                gradient_problem = "HAVE Problems"
                break
            it.iternext()

        logging.info("Finish to check, gradients:%s" % gradient_problem)
        return

    def check_layer_params(self, obj, x, check_params=None):
        """
        Checking gradients on parameters of given layer
        obj: Layer
            Object to check. obj must provide its params, param_names, forward,
            backprop. GradientChecker will use forward function and construct a
            simple loss function (sum function) to test obj's gradients.
        x: numpy.ndarray
            x is the input data of obj
        check_params: list of string
            Parameter names in check_params will be checked. If it is None, all
            parameters will be checked
        """

        logging.info("To check the gradients on parameters")

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
