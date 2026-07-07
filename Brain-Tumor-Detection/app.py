import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from PIL import Image, ImageOps
import numpy as np


st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

st.title("🧠 NeuroScan: Brain Tumor Detection")
st.markdown("Upload a patient's MRI scan to initialize the CNN diagnostic model.")


@st.cache_resource
def load_my_model():
    
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=out)
    
    
    model.load_weights('brain_tumor_final.keras')
    return model


try:
    model = load_my_model()
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("Please upload an MRI image to begin analysis.")
else:
    
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    # 4. Prediction Logic
    if st.button("Run AI Diagnostic 🚀"):
        with st.spinner("Analyzing spatial features..."):
            
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.LANCZOS)
            img_array = np.asarray(image_resized) / 255.0
            img_reshape = img_array[np.newaxis, ...]
            
        
            prediction = model.predict(img_reshape)
            
            result_index = np.argmax(prediction[0])
            result = class_names[result_index]
            confidence = prediction[0][result_index] * 100
            
    
            st.markdown("---")
            if result == "No Tumor":
                st.success(f"✅ Result: **{result}** ({confidence:.2f}% Confidence)")
            else:
                st.error(f"⚠️ Result: **{result} Detected** ({confidence:.2f}% Confidence)")
            
            st.warning("**Disclaimer:** This is an AI-assisted tool for educational purposes. Please consult a radiologist for an official medical diagnosis.")