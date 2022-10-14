import math

import layers_keras
import numpy as np
import tensorflow as tf

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """

    def call(self, inputs, training=False):
        ## If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        ## Otherwise, manually compute convolution at inference.
        ## Doesn't have to be differentiable. YAY!
        bn, h_in, w_in, c_in = inputs.shape  ## Batch #, height, width, # channels in input
        c_out = self.filters                 ## channels in output
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride

        # Cleaning padding input.
        if self.padding == "SAME":
            ph = (fh - 1) // 2
            pw = (fw - 1) // 2
        elif self.padding == "VALID":
            ph, pw = 0, 0
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")

        ## TODO: Convolve filter from above with the inputs.
        ## Note: Depending on whether you used SAME or VALID padding,
        ## the input and output sizes may not be the same

        ## Pad input if necessary

        ## Calculate correct output dimensions

        ## Iterate and apply convolution operator to each image

        ## PLEASE RETURN A TENSOR using tf.convert_to_tensor(your_array, dtype=tf.float32)
        return None
