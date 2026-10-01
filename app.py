import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image  # for opening photos

# ---------- Loading ----------

# cache_resource loads the model once, not on every click (much faster)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('banana_model.keras')

model = load_model()

# the class names saved by train.py, in the same order the model uses
with open('class_names.json') as f:
    class_names = json.load(f)

# ---------- Advice for each class ----------

advice = {
    'unripe':   ("Not yet! Wait about 5-7 days.", "🟢"),
    'ripe':     ("Wait 2-3 more days until it has brown spots.", "🟡"),
    'overripe': ("Perfect! Make banana bread today.", "🟤"),
    'rotten':   ("Too far gone, better toss it.", "⚫"),
}

# ---------- Page ----------
st.title('Banana Detector for Banana Bread')

uploaded_file = st.file_uploader(
    "Take a photo of your banana and upload it here!",
    type=['jpg', 'jpeg', 'png']  # only accept images
)

if uploaded_file:
    st.image(uploaded_file, caption="your banana")
    img = Image.open(uploaded_file)

    # ---------- Preprocessing ----------
    img = img.convert('RGB')                  # force 3 colour channels
    img = img.resize((256, 256))              # same size as in train.py
    img_array = np.array(img, dtype='float32')  # image -> numbers, shape (256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim -> (1, 256, 256, 3)
    # NOTE: no "/ 255" here, because training used raw 0-255 values

    # ---------- Prediction ----------
    predictions = model.predict(img_array)    # shape (1, 4): one probability per class
    probs = predictions[0]                    # the 4 probabilities
    best_index = int(np.argmax(probs))        # position of the highest one
    label = class_names[best_index]           # position -> class name
    confidence = float(probs[best_index]) * 100

    # ---------- Show result ----------
    message, emoji = advice.get(label, ("Unknown stage.", "❓"))
    st.subheader(f"{emoji} {label.capitalize()} ({confidence:.0f}% sure)")
    st.write(message)

    # optional: show all probabilities as a bar chart
    st.bar_chart(dict(zip(class_names, probs.tolist())))