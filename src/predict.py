import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'flower_cnn.keras')
IMG_SIZE = (150, 150)
CLASS_NAMES = ['matahari', 'mawar', 'melati', 'tulip']
CLASS_INDONESIAN = {
    'matahari': 'Bunga Matahari',
    'mawar': 'Bunga Mawar',
    'melati': 'Bunga Melati',
    'tulip': 'Bunga Tulip'
}

model = None

def load_model_once():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model tidak ditemukan di {MODEL_PATH}. Silakan train model terlebih dahulu.')
        model = load_model(MODEL_PATH)
    return model

def predict_image(image_path):
    model = load_model_once()

    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class_idx])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    predicted_name = CLASS_INDONESIAN[predicted_class]

    confidence_per_class = {
        CLASS_INDONESIAN[name]: float(conf)
        for name, conf in zip(CLASS_NAMES, predictions)
    }

    return {
        'predicted_class': predicted_class,
        'predicted_name': predicted_name,
        'confidence': confidence,
        'confidence_per_class': confidence_per_class,
        'raw_predictions': predictions.tolist()
    }

def predict_batch(image_paths):
    return [predict_image(path) for path in image_paths]

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        print(f'Predicted: {result["predicted_name"]}')
        print(f'Confidence: {result["confidence"]:.4f}')
        print('\nConfidence per class:')
        for name, conf in result['confidence_per_class'].items():
            bar = '#' * int(conf * 50)
            print(f'{name:20s}: {conf:.4f} {bar}')
    else:
        print('Usage: python -m src.predict <path_to_image>')
