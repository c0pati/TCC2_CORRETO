import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import io

# Função para carregar e pré-processar imagens do upload
def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    return img_array, np.array(img)

# Função para aplicar equalização de histograma no canal Y
def equalize_histogram_yuv(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_eq_rgb = cv2.cvtColor(img_eq_bgr, cv2.COLOR_BGR2RGB)
    return img_eq_rgb

# Função para desenhar os histogramas RGB
def plot_histogram(img, title='Histograma'):
    fig, ax = plt.subplots()
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_title(title)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Intensidade")
    ax.set_ylabel("Frequência")
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# Carregar o modelo VGG19
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg19_output = GlobalAveragePooling2D()(vgg19.output)
x = Dense(256, activation='relu')(vgg19_output)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=vgg19.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Carregar pesos treinados
model.load_weights('best_vgg19_model.keras')  # Ajuste o caminho se necessário

# Interface Streamlit
st.title("Classificação de Etiqueta: Nítida ou Borrada")
uploaded_file = st.file_uploader("Faça o upload de uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Imagem enviada:")
    preprocessed_image, original_image = preprocess_uploaded_image(uploaded_file)
    preprocessed_image_batch = np.expand_dims(preprocessed_image, axis=0)

    # Classificação
    prediction = model.predict(preprocessed_image_batch)[0][0]
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    result = 'Nítida' if prediction > 0.5 else 'Borrada'

    # Equalização
    equalized_image = equalize_histogram_yuv(original_image)

    # Exibir imagens
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Imagem Original", use_container_width=True)
    with col2:
        st.image(equalized_image, caption="Imagem Equalizada (YUV)", use_container_width=True)

    # Histogramas
    st.subheader("Histogramas RGB")
    col3, col4 = st.columns(2)
    with col3:
        hist_buf_original = plot_histogram(original_image, "Histograma Original")
        st.image(hist_buf_original, caption="Histograma da Imagem Original", use_container_width=True)
    with col4:
        hist_buf_eq = plot_histogram(equalized_image, "Histograma Equalizado")
        st.image(hist_buf_eq, caption="Histograma da Imagem Equalizada", use_container_width=True)

    # Resultado final
    st.subheader("Resultado da Classificação")
    st.success(f"Classificação: **{result}** com **{confidence:.2f}%** de confiança.")
