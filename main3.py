from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

NUM_ITERATIONS = 10
WEIGHT_CONTENT = 0.025
WEIGHT_STYLE = 1.0
TOTAL_VARIATION_WEIGHT = 1.0
SAVE_RESULTS_PATH = 'output/'

path_content = 'subject.jpg'
path_style = 'geometric.jpg'

FEATURE_LAYERS = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

RESULTING_IMG_DIMS = (400, 400)

def load_images(content_path, style_path):
    content = K.variable(preprocess_image(base_image_path))
    style = K.variable(preprocess_image(style_reference_image_path))
    return (content, style)

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(RESULTING_IMG_DIMS[0], RESULTING_IMG_DIMS[1]))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def load_Model(content, style):
    input_data = K.concatenate([content,style], axis=0)
    mod =  vgg19.VGG19(input_tensor=input_data, weights='imagenet', include_top=False)
    outputs = dict([(layer.name, layer.output) for layer in model.layers])
    return (mod, outputs)


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, 3, img_nrows, img_ncols))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def run_NN(num_steps):
    evaluator = Evaluator()
    for i in rage(num_steps):
        img_tensor, min_val, info = fmin_l_bfgs_b(evaluator.loss, img_tensor.flatten(),
                                     fprime=evaluator.grads, maxfun=20)


if __name__ == '__main__':

    c, s = load_images(path_content, path_style)
    model, outputs = load_model(c,s)

