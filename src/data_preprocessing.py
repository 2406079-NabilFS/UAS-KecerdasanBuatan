import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset_split', 'train')
VAL_PATH = os.path.join(BASE_DIR, 'dataset_split', 'validation')
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
CLASS_NAMES = ['matahari', 'mawar', 'melati', 'tulip']
CLASS_LABELS = {name: idx for idx, name in enumerate(CLASS_NAMES)}
VAL_SPLIT = 0.2
RANDOM_SEED = 42

def split_dataset():
    if os.path.exists(TRAIN_PATH):
        return

    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        train_imgs, val_imgs = train_test_split(
            images, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )

        train_class_dir = os.path.join(TRAIN_PATH, class_name)
        val_class_dir = os.path.join(VAL_PATH, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(class_path, img), os.path.join(val_class_dir, img))

    print(f'Dataset split complete.')
    for class_name in CLASS_NAMES:
        train_count = len(os.listdir(os.path.join(TRAIN_PATH, class_name)))
        val_count = len(os.listdir(os.path.join(VAL_PATH, class_name)))
        print(f'  {class_name}: {train_count} train, {val_count} val')

def get_data_generators():
    split_dataset()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=RANDOM_SEED
    )

    return train_generator, val_generator

def load_data():
    split_dataset()

    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []

    for class_name in CLASS_NAMES:
        label = CLASS_LABELS[class_name]

        train_class_dir = os.path.join(TRAIN_PATH, class_name)
        for img_name in os.listdir(train_class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = load_img(os.path.join(train_class_dir, img_name), target_size=IMG_SIZE)
                x_train_list.append(img_to_array(img) / 255.0)
                y_train_list.append(label)

        val_class_dir = os.path.join(VAL_PATH, class_name)
        for img_name in os.listdir(val_class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = load_img(os.path.join(val_class_dir, img_name), target_size=IMG_SIZE)
                x_val_list.append(img_to_array(img) / 255.0)
                y_val_list.append(label)

    x_train = np.array(x_train_list)
    y_train = to_categorical(np.array(y_train_list), num_classes=len(CLASS_NAMES))
    x_val = np.array(x_val_list)
    y_val = to_categorical(np.array(y_val_list), num_classes=len(CLASS_NAMES))

    return (x_train, y_train), (x_val, y_val)

def visualize_samples(num_samples=12):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    sample_count = 0
    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            continue
        img_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for i in range(min(num_samples // len(CLASS_NAMES), len(img_files))):
            if sample_count >= num_samples:
                break
            img_path = os.path.join(class_path, img_files[i])
            img = load_img(img_path, target_size=IMG_SIZE)
            axes[sample_count].imshow(img)
            axes[sample_count].set_title(class_name.title(), fontsize=12, pad=8)
            axes[sample_count].axis('off')
            sample_count += 1

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, 'dataset_samples.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    return fig

if __name__ == '__main__':
    visualize_samples()
    train_gen, val_gen = get_data_generators()
    print(f'Training samples: {train_gen.samples}')
    print(f'Validation samples: {val_gen.samples}')
    print(f'Class mapping: {train_gen.class_indices}')
