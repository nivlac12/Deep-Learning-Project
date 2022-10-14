from typing import Iterable

import tensorflow as tf

################################################################################################
## Input handling functions


def init_regularizer(reg):
    # Return deserialization of reg function if one is given; 0 function otherwise
    if reg is None:
        return lambda x: tf.constant(0)
    return tf.keras.regularizers.get(reg)()


def init_constraint(con):
    # Return deserialization of constraint function if one is given; identity function otherwise
    if con is None:
        return lambda x: x
    return tf.keras.constraints.get(con)()


def init_activation(act):
    # Return deserialization of activation function if one is given; identity function otherwise
    if act is None:
        return lambda x: x
    return tf.keras.activations.get(act)


def ensure_tuple(x, dims=2, default=None):
    ## Make sure that x is a tuple with dims entries. Either duplicate dims times or subset.
    ## If length is too short, add default value to end of list until dims requirement met.
    if not isinstance(x, Iterable):
        return tuple([x] * dims)
    if len(x) >= dims:
        return x[:dims]
    if default is None:
        default = x[0]
    return tuple(x + [default] * (dims - len(x)))


################################################################################################


class Conv2D(tf.keras.layers.Layer):
    """
    Performs convolution on the call inputs.
    Refer to the tensorflow Conv2D documentation for more details.
    Method performs similarly on training and inference.
    Optimization done by tf.Variable, so... no gradient logic necessary.

    NOTE: all of the parameters listed below will need to be used somewhere within
    your build or call functions

    :param filters: Number of output filters in the convolution
    :param kernel_size: Length-2 tuple specifying height and width of 2D convolution window
    :param strides: Length-2 tuple specifying strides of convolution along height and width
    :param padding: "valid" means no padding, "same" results in padding with zeros evenly to
                    left/right or up/down of input
    :param activation: Activation function to use
    :param use_bias: Boolean indicating whether to use a bias vector
    :param kernel_initializer: Distribution used to initialize the kernel weights matrix
    :param bias_initializer: Distribution used to intialize the bias vector
    :param kernel_constraint: Imposes constraints on kernel weight values (ex: can limit kernel to
                              only positive values by passing non_neg)
    :param bias_constraint: Imposes constraints on bias values
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        # data_format=None,      ## For other input specifications: (b, c, h, w), etc.
        # dilation_rate=(1, 1),  ## For sparce convolution support
        # groups=1,              ## Advanced input/output relationships
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters

        ## If somebody doesn't give a length-2 tuple, make it a length-2 tuple
        ## tf.nn.conv wants upper-case, but Conv2D accepts both... so just make it upper
        ## If activation is None, just make it an identity function f(x) = x
        self.kernel_size = ensure_tuple(kernel_size)
        self.strides     = ensure_tuple(strides)
        self.padding     = padding.upper()
        self.activation  = init_activation(activation)
        self.use_bias    = use_bias

        ## If somebody says initializer is none, this will lead to an error later on
        ## tf.keras.initializers.get returns the appropriate initializer class
        ## given a string (i.e. glorot_uniform => tf.keras.initializers.glorotuniform)
        self.kernel_initializer   = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer     = tf.keras.initializers.get(bias_initializer)

        ## If regularization is None, just make it a lambda function that returns 0
        self.kernel_regularizer   = init_regularizer(kernel_regularizer)
        self.bias_regularizer     = init_regularizer(bias_regularizer)
        self.activity_regularizer = init_regularizer(activity_regularizer)

        ## If constraint is None, just make it an identity function
        self.kernel_constraint    = init_constraint(kernel_constraint)
        self.bias_constraint      = init_constraint(bias_constraint)

        ## We don't have the shapes for kernel and bias yet...
        self.kernel = None
        self.bias   = None

    def build(self, input_shape):
        ## TODO 1: Initialize your kernel weights and your bias vector
        ## Refer to the __call__ method at
        ## https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Initializer
        ## for the arguments you can pass in

    def call(self, inputs, training=False):
        ## Constrain your kernel and bias (refer to docstring for what this means)
        self.kernel = self.kernel_constraint(self.kernel)
        self.bias   = self.bias_constraint(self.bias)

        ## TODO 2: Perform convolution using tf.nn.convolution and bias and activation if necessary

        ## TODO 3: use self.add_loss on the result of applying your regularizers
        ## to tell tensorflow that the regularizers are going to affect the loss

        return None


################################################################################################


class BatchNormalization(tf.keras.layers.Layer):
    """
    BatchNormalization Layer

    - During Training:
        - Calculate average/variance for the current batch.
        - Update moving averages (note; these are stat aggregations, NOT trainable variables).
            - So as your code trains, the average will update to reflect more recent batches.
            - By end of training, the moving mean/var will be representative of the training set.
            - Batch/moving stats are computed with respect to # of input channels.
        - Make sure to incorporate trainable beta and gamma parameters if needed (center/scale).
        - Perform actual batch normalization using Keras assumptions (see docs).

    - During Inference:
        - Use accumulated moving averages instead of current batch statistics.
        - Do not update moving averages. This should only be updated in training.

    - In Both:
        - Regularize w.r.t. beta/gamma if function is provided

    - Useful Methods:
        - tf.nn.moments
        - tf.Variable()
        - tf.Variable.assign
        - tf.sqrt

    NOTE: all of the parameters listed below will need to be used somewhere within
    your build or call functions

    :param axis: Integer specifying axis that should be normalized
    :param momentum: Momentum for moving average
    :param epsilon: Small float added to variance to avoid dividing by zero
    :param center: If True, add offset of beta to normalized tensor. If False, beta is ignored
    :param scale: If True, multiply by gamma. If False, gamma is not used.
    :param beta_initializer: Initializer for the beta weight
    :param gamma_initializer: Initializer for the gamma weight
    :param moving_mean_initializer: Initializer for the moving mean
    :param moving_var_initializer: Initializer for the moving variance
    :param beta_regularizer: Optional regularizer for the beta weight
    :param gamma_regularizer: Optional regularizer for the gamma weight
    :param beta_constraint: Imposes constraints on beta value
    :param gamma_constraint: Imposes constraints on gamma value
    """

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Axis dictates the type of normalization (batch norm, layer norm, instance norm)
        self.axis     = axis
        self.momentum = momentum
        self.epsilon  = epsilon
        self.center   = center
        self.scale    = scale

        # Initializes beta, gamma, moving mean, and moving variance
        self.beta_initializer  = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(
            moving_mean_initializer
        )
        self.moving_variance_initializer = tf.keras.initializers.get(
            moving_variance_initializer
        )

        # Initializes beta and gamma regularizers and constraints
        self.beta_regularizer  = init_regularizer(beta_regularizer)
        self.gamma_regularizer = init_regularizer(gamma_regularizer)
        self.beta_constraint   = init_constraint(beta_constraint)
        self.gamma_constraint  = init_constraint(gamma_constraint)

    def build(self, input_shape):
        ## TODO 1: Initialize self.<> using self.<>_initializer. These may need to be either 0 or 1
        ## depending on self.center and self.scale (take a look at the docstring).
        ## Make sure you understand the purpose self.axis is playing.

        ## TODO 2: Initialize components as tf.Variables; specify trainable status as appropriate

    def call(self, inputs, training=False):
        ## Apply the constraints. An example of using a deserialized function
        self.beta  = self.beta_constraint(self.beta)
        self.gamma = self.gamma_constraint(self.gamma)

        ## TODO 3: Implement forward pass. This involves calculating batch average
        ## and variance, updating your moving average and variance, and actually
        ## normalizing your inputs.

        ## tf.nn.moments computes the batch average and batch variance for you
        if training:
            bmu, bvar = tf.nn.moments(inputs, range(len(inputs.shape) - 1))

        ## TODO 4: use self.add_loss on the result of applying your regularizers
        ## to tell tensorflow that the regularizers are going to affect the loss
        return None


################################################################################################


class Dropout(tf.keras.layers.Layer):
    """
    Dropout Layer

    - During Training:
        - Randomly select rate% of the the inputs to zero out and scale appropriately.
        - Abide by random seed as specified.
        - Abide by noise_shape if specified; otherwise, let noise_shape be shape of entries.

    - During Inference:
        - Allow inputs to pass through unchanged
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=False):
        ## TODO: Implement forward pass for dropout
        return None
