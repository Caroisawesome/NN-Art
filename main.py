from keras import backend as K
import tensorflow as tf
#import tensorflow.python.keras.backend as K
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

cImPath = './geometric.jpg'
sImPath = './subject.jpg'
genImOutputPath = 'output.jpg'

targetHeight = 300
targetWidth = 300
targetSize = (targetHeight, targetWidth)

def prepare_img(img_name, w, h):
    t_size = (h, w)
    img = load_img(path=img_name, target_size=t_size)
    img_arr = img_to_array(img)
    expand_img = np.expand_dims(img_arr, axis=0)
    return preprocess_input(expand_img)

def create_model(layers, model):
  outputs = [model.get_layer(name).output for name in layers]
  model = tf.keras.models.Model([model.input], outputs)
  return model

# Style Loss
# def get_Gram_matrix(F):
#      G = K.dot(F, K.transpose(F))
#      return G

def get_Gram_matrix(tensor):
  temp = tensor
  temp = tf.squeeze(temp)
  fun = tf.reshape(temp, [temp.shape[2], temp.shape[0]*temp.shape[1]])
  result = tf.matmul(temp, temp, transpose_b=True)
  gram = tf.expand_dims(result, axis=0)
  return gram

class Custom_Style_Model(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers, mdl):
    super(Custom_Style_Model, self).__init__()
    self.vgg =  mdl
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    # Scale back the pixel values
    inputs = inputs*255.0
    # Preprocess them with respect to VGG19 stats
    preprocessed_input = preprocess_input(inputs)
    # Pass through the mini network
    outputs = self.vgg(preprocessed_input)
    # Segregate the style and content representations
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    # Calculate the gram matrix for each layer
    style_outputs = [get_Gram_matrix(style_output)
                     for style_output in style_outputs]

    # Assign the content representation and gram matrix in
    # a layer by layer fashion in dicts
    content_dict = {content_name:value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content':content_dict, 'style':style_dict}

# Content Loss
def get_feature_reps(x, layer_names, model):
    """
    Get feature representations of input x for one or more layers in a given model.
    """
    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(session=tf_session)
        N_l = featRawShape[-1]
        M_l = featRawShape[1]*featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices

def get_content_loss(F, P):
    cLoss = 0.5*K.sum(K.square(F - P))
    return cLoss


# The loss function to optimize
# Total Loss = alpha(contentLoss) + beta(styleLoss)
def total_loss(outputs, num_style_layers, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([style_weights[name]*tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    # Normalize
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    # Normalize
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss


# def get_style_loss(ws, Gs, As):
#     sLoss = K.variable(0.)
#     for w, G, A in zip(ws, Gs, As):
#         M_l = K.int_shape(G)[1]
#         N_l = K.int_shape(G)[0]
#         G_gram = get_Gram_matrix(G)
#         A_gram = get_Gram_matrix(A)
#         sLoss+= w*0.25*K.sum(K.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
#     return sLoss


# def get_total_loss(gImPlaceholder, alpha=1.0, beta=10000.0):
#     F = get_feature_reps(gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]
#     Gs = get_feature_reps(gImPlaceholder, layer_names=sLayerNames, model=gModel)
#     contentLoss = get_content_loss(F, P)
#     styleLoss = get_style_loss(ws, Gs, As)
#     totalLoss = alpha*contentLoss + beta*styleLoss
#     return totalLoss

# def calculate_loss(gImArr):
#     """
#     Calculate total loss using K.function
#     """
#     if gImArr.shape != (1, targetWidth, targetWidth, 3):
#         gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
#     loss_fcn = K.function([gModel.input], [get_total_loss(gModel.input)])
#     return loss_fcn([gImArr])[0].astype('float64')

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    # Extract the features
    outputs = extractor(image)
    # Calculate the loss
    loss = total_loss(outputs, 4, 1)
  # Determine the gradients of the loss function w.r.t the image pixels
  grad = tape.gradient(loss, image)
  # Update the pixels
  opt.apply_gradients([(grad, image)])
  # Clip the pixel values that fall outside the range of [0,1]
  image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# def get_grad(gImArr):
#     """
#     Calculate the gradient of the loss function with respect to the generated image
#     """
#     if gImArr.shape != (1, targetWidth, targetHeight, 3):
#         gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
#         grad_fcn = K.function([gModel.input], 
#                               K.gradients(get_total_loss(gModel.input), [gModel.input]))
#         grad = grad_fcn([gImArr])[0].flatten().astype('float64')
#     return grad



content_img = prepare_img(cImPath, targetWidth, targetHeight)
print("Content Img", content_img.shape)

style_img = prepare_img(sImPath, targetWidth, targetHeight)


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

c_layers = 'block4_conv2'
s_layers = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                ]

layers = [
    'block4_conv2',
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
]


mdl = create_model(layers, vgg)
extractor = Custom_Style_Model(s_layers, c_layers, mdl)
style_targets = extractor(style_img)['style']
content_targets = extractor.call(content_img)['content']

opt = tf.optimizers.Adam(learning_rate=0.02)

# Custom weights for style and content updates
style_weight=100
content_weight=10

# Custom weights for different style layers
style_weights = {'block1_conv1': 1.,
                 'block2_conv1': 0.8,
                 'block3_conv1': 0.5,
                 'block4_conv1': 0.3,
                 'block5_conv1': 0.1}


## TRAIN
epochs = 40
steps_per_epoch = 100

target_image = tf.Variable(content_img)

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(target_image)
  plt.imshow(np.squeeze(target_image.read_value(), 0))
  plt.title("Train step: {}".format(step))
  plt.show()
