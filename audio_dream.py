#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
    raw_audio_music_preprocessing.py

!pip install tensorflow_io

!wget https://github.com/dcl-boop/neural_synthesizer/raw/main/m5_50ep_4r10.h5
!wget https://github.com/dcl-boop/neural_synthesizer/raw/main/cut_snippet_classical.wav
"""
import os
import pickle

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
import numpy as np

import matplotlib as mpl

import io

import soundfile as sf




def calc_loss(audio, model):
    """ Calculate loss

        The loss is the sum of the activations in the chosen layers. 
        The loss is normalized at each layer so the contribution from larger 
        layers does not outweigh smaller layers. Normally, loss is a quantity 
        you wish to minimize via gradient descent. In DeepDream, you will 
        maximize this loss via gradient ascent.
    """
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    audio_batch = tf.expand_dims(audio, axis=0)
    layer_activations = model(audio_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)

"""## Gradient ascent

Once you have calculated the loss for the chosen layers, all that is left is to calculate the gradients with respect to the image, and add them to the original image. 

Adding the gradients to the image enhances the patterns seen by the network. At each step, you will have created an image that increasingly excites the activations of certain layers in the network.

The method that does this, below, is wrapped in a `tf.function` for performance. It uses an `input_signature` to ensure that the function is not retraced for different image sizes or `steps`/`step_size` values. See the [Concrete functions guide](../../guide/function.ipynb) for details.
"""

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=None
  )
  def __call__(self, audio, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(audio)
          loss = calc_loss(audio, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, audio)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 
        
        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        audio = audio + gradients*step_size
        #audio = tf.clip_by_value(audio, -1, 1)

      return loss, audio


"""## Main Loop"""

def run_deep_dream_simple(audio, deepdream, steps=100, step_size=0.01):
  # Convert from uint8 to the range expected by the model.
  # audio = tf.keras.applications.inception_v3.preprocess_input(audio)
  # audio = tf.convert_to_tensor(audio)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, audio = deepdream(audio, run_steps, tf.constant(step_size))
    
    print ("Step {}, loss {}".format(step, loss))

  #result = deprocess(audio)
  return audio


def main(model,original,output_name):

    """The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in a way that the image increasingly "excites" the layers. The complexity of the features incorporated depends on layers chosen by you, i.e, lower layers produce strokes or simple patterns, while deeper layers give sophisticated features in images, or even whole objects.

    The InceptionV3 architecture is quite large (for a graph of the model architecture see TensorFlow's [research repo](https://github.com/tensorflow/models/tree/master/research/slim)). For DeepDream, the layers of  interest are those where the convolutions are concatenated. There are 11 of these layers in InceptionV3, named 'mixed0' though 'mixed10'. Using different layers will result in different dream-like images. Deeper layers respond to higher-level features (such as eyes and faces), while earlier layers respond to simpler features (such as edges, shapes, and textures). Feel free to experiment with the layers selected below, but keep in mind that deeper layers (those with a higher index) will take longer to train on since the gradient computation is deeper.
    """

    # Maximize the activations of these layers
    # FOR OUR MODEL, ONLY A SINGLE 'dense_2' LAYER IS PRESENT
    #'conv1d_2', 'conv1d_1', 'conv1d', 'conv1d_3', 'dense'
    names = ['dense']#, 'conv1d_2', 'conv1d_1', 'conv1d', 'conv1d_3'] #'dense', 'dense_1', 
    layers = [model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    deepdream = DeepDream(dream_model)

    original_audio = tfio.audio.AudioIOTensor(f'{original}.wav').to_tensor()
    original_audio = tf.cast(original_audio, tf.float32)

    dream_audio = run_deep_dream_simple(original_audio, deepdream, steps=100, step_size=0.01)
    
    da = tfio.audio.encode_wav(dream_audio,22050)

    out = { 'dream_audio': dream_audio,
            'wav': da}
    with open(output_name, 'wb') as w:
        pickle.dump(out, w)

    # Write out audio as 24bit PCM WAV
    try:
        sf.write(f'{output_name}.wav', dream_audio.numpy(), 22050, subtype='FLOAT')
        print(f"Successfully wrote {output_name}.wav")
    except Exception as e:
        print("Failed to write .wav")
        raise e


if __name__ == '__main__':

    from raw_audio_music_preprocessing import RawAudioDataPreProcess

    r = RawAudioDataPreProcess()
    
    ### LOAD OUR PRETRAINED MODEL
    base_model = tf.keras.models.load_model('saved_model/m5_30ep_22050_10_real')


    # download list of audio snippets to try

    snippets = os.listdir(os.curdir + '/snippets')
    # input_dir is where presnips are stored
    input_dir = './presnips'
    output_dir = './snippouts'

    #main(model,original,output_name)

    for original in snippets:
        print()
        print(f"Original: {original}")
        
        r.preprocess_original(original)
        
        orig = original.replace('.wav', '')

        main(base_model, f'{input_dir}/{orig}', f'{output_dir}/{orig}_dreamy')

