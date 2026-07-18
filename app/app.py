import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.predict import predict_image, CLASS_NAMES, CLASS_INDONESIAN, MODEL_PATH

st.set_page_config(
    page_title='Klasifikasi Bunga dengan CNN',
    page_icon='🌸',
    layout='wide'
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f0f2f6;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 24px;
        border-radius: 12px;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        margin: 4px 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🌸 Klasifikasi Jenis Bunga dengan CNN</h1><p>Deep Learning - Convolutional Neural Network untuk Mengenali 4 Jenis Bunga</p></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('🌻 **Matahari**')
with col2:
    st.markdown('🌹 **Mawar**')
with col3:
    st.markdown('🤍 **Melati**')
with col4:
    st.markdown('🌷 **Tulip**')

st.markdown('---')

tab1, tab2 = st.tabs(['📤 Klasifikasi Gambar', '📊 Informasi Model'])

with tab1:
    st.subheader('Upload Gambar Bunga')

    input_method = st.radio('Pilih metode input:', ['Upload File', 'Contoh Gambar'], horizontal=True)

    image = None
    image_path = None

    if input_method == 'Upload File':
        uploaded_file = st.file_uploader(
            'Pilih file gambar (JPG, JPEG, PNG)',
            type=['jpg', 'jpeg', 'png'],
            label_visibility='collapsed'
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_path = uploaded_file
    else:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
        sample_images = []
        for class_name in CLASS_NAMES:
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                for f in os.listdir(class_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_images.append(os.path.join(class_path, f))

        if sample_images:
            cols = st.columns(4)
            selected_sample = None
            for idx, img_path in enumerate(sample_images[:12]):
                col_idx = idx % 4
                with cols[col_idx]:
                    img = Image.open(img_path)
                    img.thumbnail((150, 150))
                    class_name = os.path.basename(os.path.dirname(img_path))
                    if st.button(f'{CLASS_INDONESIAN.get(class_name, class_name)}', key=f'btn_{idx}'):
                        selected_sample = img_path
                    st.image(img, width=120)

            if selected_sample:
                image = Image.open(selected_sample)
                image_path = selected_sample

    if image is not None:
        st.markdown('### Hasil Klasifikasi')

        result_col, viz_col = st.columns([1, 1])

        with result_col:
            st.image(image, caption='Gambar Input', use_container_width=True)

            with st.spinner('Menganalisis gambar...'):
                if hasattr(image_path, 'name'):
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        image.save(tmp.name)
                        result = predict_image(tmp.name)
                    os.unlink(tmp.name)
                else:
                    result = predict_image(image_path)

            st.markdown(
                f'<div class="prediction-card">'
                f'<h2 style="color: #764ba2; margin: 0;">{result["predicted_name"]}</h2>'
                f'<p style="font-size: 1.2rem; margin: 0.5rem 0;">'
                f'Tingkat Keyakinan: <strong>{result["confidence"]*100:.1f}%</strong></p>'
                f'</div>',
                unsafe_allow_html=True
            )

        with viz_col:
            st.subheader('Tingkat Keyakinan per Kelas')

            conf_df = pd.DataFrame([
                {'Kelas': name, 'Confidence': conf}
                for name, conf in result['confidence_per_class'].items()
            ])
            conf_df = conf_df.sort_values('Confidence', ascending=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(conf_df['Kelas'], conf_df['Confidence'], color=['#9b59b6', '#3498db', '#2ecc71', '#f39c12'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence')
            ax.set_title('Distribusi Keyakinan Model', fontweight='bold')

            for bar, val in zip(bars, conf_df['Confidence']):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val*100:.1f}%', va='center', fontsize=11, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

with tab2:
    st.subheader('Arsitektur Model (Transfer Learning - MobileNetV2)')

    st.markdown("""
    | Layer | Detail |
    |-------|--------|
    | **Input** | Gambar 150x150x3 (RGB) |
    | **Backbone** | MobileNetV2 (ImageNet weights, frozen) |
    | **Head** | GlobalAveragePooling2D → BatchNorm → Dropout(30%) → Dense(128) → BatchNorm → Dropout(30%) → Dense(4, Softmax) |
    """)

    st.subheader('Detail Dataset')
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    for idx, (col, class_name) in enumerate(zip([col_d1, col_d2, col_d3, col_d4], CLASS_NAMES)):
        class_path = os.path.join(dataset_path, class_name)
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(class_path) else 0
        with col:
            st.metric(label=CLASS_INDONESIAN[class_name], value=f'{count} gambar')

    st.info('''
    **Total Dataset**: 385 gambar | **Train/Val Split**: 80%/20% | **Image Size**: 150x150 RGB
    ''')

    st.subheader('Teknik Preprocessing & Augmentasi')
    st.markdown('''
    - **Rescaling**: Normalisasi pixel ke [0,1]
    - **Rotation Range**: ±40°
    - **Width/Height Shift**: 20%
    - **Shear Range**: 20%
    - **Zoom Range**: 20%
    - **Horizontal Flip**: Ya
    - **Fill Mode**: Nearest
    ''')

    st.subheader('Hyperparameter')
    st.markdown('''
    - **Optimizer**: Adam (lr=0.001)
    - **Loss Function**: Categorical Crossentropy
    - **Batch Size**: 32
    - **Epochs**: 100 (dengan Early Stopping)
    - **Regularization**: L2 (0.001) + Dropout
    - **Learning Rate Scheduling**: ReduceLROnPlateau
    ''')

model_exists = os.path.exists(MODEL_PATH)
if not model_exists:
    st.warning('⚠️ Model belum di-training. Jalankan `python -m src.train` terlebih dahulu.')

st.markdown('---')
st.markdown(
    '<div class="footer">'
    'Tugas Besar UAS - Kecerdasan Buatan | '
    'Implementasi Computer Vision untuk Klasifikasi Jenis Bunga '
    'Menggunakan Deep Learning Berbasis CNN'
    '</div>',
    unsafe_allow_html=True
)
