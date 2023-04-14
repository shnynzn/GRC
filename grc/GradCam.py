# Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model

class GradCAM:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName
        
        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

        # record operations for automatic differentiation
    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output]) 
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        
        # Normalize the heatmap
        cam3 = tf.maximum(cam3, 0) / tf.math.reduce_max(cam3) 

        return cam3.numpy()

def overlay_gradCAM(img, cam3, output_path="grad_cam_image.jpg", alpha=0.4):
    img= image.img_to_array(img)
    cam3 = np.uint8(255 * cam3)  # Back scaling to 0-255 from 0 - 1
    #cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    jet = c_map.get_cmap("jet") # Colorizing heatmap
    jet_colors = jet(np.arange(256))[:, :3] # Using RGB values
    jet_cam3 = jet_colors[cam3]
    jet_cam3 = image.array_to_img(jet_cam3)
    jet_cam3 = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_cam3 = image.img_to_array(jet_cam3c)
    
    #new_img = 0.3 * cam3 + 0.5 * img
    
    new_img = jet_heatmap * alpha + img # Superimposing the heatmap on original image
    new_img = img_to_array(new_img)
    
    new_img.save(output_path) # Saving the superimposed image
    display(Image(output_path)) # Displaying Grad-CAM Superimposed Image
    
    #return (new_img * 255.0 / new_img.max()).astype("uint8")
