import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# LOAD MODELS 
cnn_model = load_model("brain_tumor_model.keras", compile=False)

# CLASS LABELS 
classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

# FEATURE EXTRACTOR 
feature_extractor = Model(
    inputs=cnn_model.input,
    outputs=cnn_model.layers[-2].output
)

# GRAD-CAM 
def get_gradcam(model, img_array):

    # find last conv layer safely
    last_conv_layer = None

    for layer in reversed(model.layers):
        if "conv" in layer.name or "Conv" in layer.__class__.__name__:
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("No convolution layer found in model")

    grad_model = Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()
def overlay_heatmap(img, heatmap):

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
st.title("🧠 Brain Tumor Detection System")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded:

    img = image.load_img(uploaded, target_size=(224, 224))
    st.image(img, caption="Uploaded MRI")

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # CNN prediction
    pred = cnn_model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    # Grad-CAM
    heatmap = get_gradcam(cnn_model, img_array)
    result_img = overlay_heatmap(img, heatmap)

    # Output
    st.subheader("📊 Results")
    st.write("Class:", classes[class_idx])
    st.write("Confidence:", float(confidence))

    st.subheader("🔥 Heatmap")
    st.image(result_img)