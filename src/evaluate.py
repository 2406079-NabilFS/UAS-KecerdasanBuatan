import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.data_preprocessing import get_data_generators, CLASS_NAMES

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'flower_cnn.keras')
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def evaluate_model():
    print('Loading model...')
    model = load_model(MODEL_PATH)

    _, val_generator = get_data_generators()
    print(f'Validation samples: {val_generator.samples}')

    print('Evaluating on validation data...')
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    print('Generating predictions...')
    val_generator.reset()
    steps = val_generator.samples // val_generator.batch_size
    if val_generator.samples % val_generator.batch_size != 0:
        steps += 1
    predictions = model.predict(val_generator, steps=steps, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes[:len(y_pred)]

    print('\nClassification Report:')
    print('=' * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Confusion matrix saved to: {save_path}')

    for i, class_name in enumerate(CLASS_NAMES):
        mask = y_true == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f'{class_name}: {class_acc:.4f}')

    return accuracy

if __name__ == '__main__':
    evaluate_model()
