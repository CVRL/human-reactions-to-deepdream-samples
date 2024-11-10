# boilerplate code
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import IPython.display
from PIL import Image

#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()


def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, img0=img_noise,
                     iter_n=10, step=4.0, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
    return img


# parameters
model_fn = 'tensorflow_inception_graph.pb' 
saved_image_data_format = 'channels_last'
dataset_mean = 117.0 # imagenet_mean = 117.0

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# graph_def from a pb file
# graph_def - a GraphDef proto containing operations to be imported into the default graph.
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


# define the input tensor    
t_input = tf.placeholder(np.float32, name='input')
t_preprocessed = tf.transpose(t_input, perm=[2,0,1]) if saved_image_data_format == 'channels_first' else t_input
t_preprocessed = tf.expand_dims(t_preprocessed-dataset_mean, 0)

# input_map - a dictionary mapping input names (as strings) in graph_def to Tensor objects. 
# The values of the named input tensors in the imported graph will be re-mapped to the respective Tensor values.
# we assume that the name of the input tensor is "input"
input_map = {'input':t_preprocessed}

# A list of strings containing operation names in graph_def that will be returned as Operation objects; and/or tensor names in graph_def that will be returned as Tensor objects.
tf.import_graph_def(graph_def, input_map)


# Layers within the InceptionV3 model which can be used to generate Deep Dream images
layers = ['conv2d0_pre_relu', 'conv2d0', 'maxpool0', 'localresponsenorm0', 'conv2d1_pre_relu', 'conv2d1', 'conv2d2_pre_relu', 'conv2d2', 'localresponsenorm1', 'maxpool1', 'mixed3a_1x1_pre_relu', 'mixed3a_1x1', 'mixed3a_3x3_bottleneck_pre_relu', 'mixed3a_3x3_bottleneck', 'mixed3a_3x3_pre_relu', 'mixed3a_3x3', 'mixed3a_5x5_bottleneck_pre_relu', 'mixed3a_5x5_bottleneck', 'mixed3a_5x5_pre_relu', 'mixed3a_5x5', 'mixed3a_pool', 'mixed3a_pool_reduce_pre_relu', 'mixed3a_pool_reduce', 'mixed3a', 'mixed3b_1x1_pre_relu', 'mixed3b_1x1', 'mixed3b_3x3_bottleneck_pre_relu', 'mixed3b_3x3_bottleneck', 'mixed3b_3x3_pre_relu', 'mixed3b_3x3', 'mixed3b_5x5_bottleneck_pre_relu', 'mixed3b_5x5_bottleneck', 'mixed3b_5x5_pre_relu', 'mixed3b_5x5', 'mixed3b_pool', 'mixed3b_pool_reduce_pre_relu', 'mixed3b_pool_reduce', 'mixed3b', 'maxpool4', 'mixed4a_1x1_pre_relu', 'mixed4a_1x1', 'mixed4a_3x3_bottleneck_pre_relu', 'mixed4a_3x3_bottleneck', 'mixed4a_3x3_pre_relu', 'mixed4a_3x3', 'mixed4a_5x5_bottleneck_pre_relu', 'mixed4a_5x5_bottleneck', 'mixed4a_5x5_pre_relu', 'mixed4a_5x5', 'mixed4a_pool', 'mixed4a_pool_reduce_pre_relu', 'mixed4a_pool_reduce', 'mixed4a', 'mixed4b_1x1_pre_relu', 'mixed4b_1x1', 'mixed4b_3x3_bottleneck_pre_relu', 'mixed4b_3x3_bottleneck', 'mixed4b_3x3_pre_relu', 'mixed4b_3x3', 'mixed4b_5x5_bottleneck_pre_relu', 'mixed4b_5x5_bottleneck', 'mixed4b_5x5_pre_relu', 'mixed4b_5x5', 'mixed4b_pool', 'mixed4b_pool_reduce_pre_relu', 'mixed4b_pool_reduce', 'mixed4b', 'mixed4c_1x1_pre_relu', 'mixed4c_1x1', 'mixed4c_3x3_bottleneck_pre_relu', 'mixed4c_3x3_bottleneck', 'mixed4c_3x3_pre_relu', 'mixed4c_3x3', 'mixed4c_5x5_bottleneck_pre_relu', 'mixed4c_5x5_bottleneck', 'mixed4c_5x5_pre_relu', 'mixed4c_5x5', 'mixed4c_pool', 'mixed4c_pool_reduce_pre_relu', 'mixed4c_pool_reduce', 'mixed4c', 'mixed4d_1x1_pre_relu', 'mixed4d_1x1', 'mixed4d_3x3_bottleneck_pre_relu', 'mixed4d_3x3_bottleneck', 'mixed4d_3x3_pre_relu', 'mixed4d_3x3', 'mixed4d_5x5_bottleneck_pre_relu', 'mixed4d_5x5_bottleneck', 'mixed4d_5x5_pre_relu', 'mixed4d_5x5', 'mixed4d_pool', 'mixed4d_pool_reduce_pre_relu', 'mixed4d_pool_reduce', 'mixed4d', 'mixed4e_1x1_pre_relu', 'mixed4e_1x1', 'mixed4e_3x3_bottleneck_pre_relu', 'mixed4e_3x3_bottleneck', 'mixed4e_3x3_pre_relu', 'mixed4e_3x3', 'mixed4e_5x5_bottleneck_pre_relu', 'mixed4e_5x5_bottleneck', 'mixed4e_5x5_pre_relu', 'mixed4e_5x5', 'mixed4e_pool', 'mixed4e_pool_reduce_pre_relu', 'mixed4e_pool_reduce', 'mixed4e', 'maxpool10', 'mixed5a_1x1_pre_relu', 'mixed5a_1x1', 'mixed5a_3x3_bottleneck_pre_relu', 'mixed5a_3x3_bottleneck', 'mixed5a_3x3_pre_relu', 'mixed5a_3x3', 'mixed5a_5x5_bottleneck_pre_relu', 'mixed5a_5x5_bottleneck', 'mixed5a_5x5_pre_relu', 'mixed5a_5x5', 'mixed5a_pool', 'mixed5a_pool_reduce_pre_relu', 'mixed5a_pool_reduce', 'mixed5a', 'mixed5b_1x1_pre_relu', 'mixed5b_1x1', 'mixed5b_3x3_bottleneck_pre_relu', 'mixed5b_3x3_bottleneck', 'mixed5b_3x3_pre_relu', 'mixed5b_3x3', 'mixed5b_5x5_bottleneck_pre_relu', 'mixed5b_5x5_bottleneck', 'mixed5b_5x5_pre_relu', 'mixed5b_5x5', 'mixed5b_pool', 'mixed5b_pool_reduce_pre_relu', 'mixed5b_pool_reduce', 'mixed5b', 'head0_pool', 'head0_bottleneck_pre_relu', 'head0_bottleneck', 'head1_pool', 'head1_bottleneck_pre_relu', 'head1_bottleneck']

# This can be used to make the directories to put the dreamified images into
for layer in layers:
    os.makedirs(f'dreamifiedImages/{layer}',exists_ok=True)
        
#loop to generate 50 iterations deep dream for each layer (Note: images did not appear to change after more than 50 iterations and thus 50 was deemed sufficient.)
for i in range(5):
    for layer in layers:
        file_to_be_dreamified_name = f"gaussianNoise800x800_{i+1}.png"
        file_to_be_dreamified = f"gaussianNoise800x800_{i+1}.png"
        img0 = PIL.Image.open(file_to_be_dreamified)
        img0 = np.float32(img0)
        print(layer)
        for j in range(50):
            img0 = render_deepdream(T(layer)[:,:,:,:], img0, iter_n=1, step=2.0, octave_n=5, octave_scale=1.4)
        layer_to_folder = layer.replace("/","-")
        fname = f'dreamifiedImages/{layer_to_folder}/'+file_to_be_dreamified_name[:-4]+f"_dreamified.png"
        a = np.uint8((np.clip(img0/255, 0, 1)*255))
        dream_savable = PIL.Image.fromarray(a)
        dream_savable.save(fname, quality = 100)