import os
import numpy as np
import matplotlib.pyplot as plt
from src.model import build_transfer_learning_model, get_callbacks, MODEL_PATH
from src.data_preprocessing import get_data_generators

EPOCHS = 50

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_history.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Training plot saved to: {save_path}')

def main():
    print('Loading data generators...')
    train_generator, val_generator = get_data_generators()

    print(f'\nBuilding Transfer Learning model (MobileNetV2)...')
    num_classes = len(train_generator.class_indices)
    model = build_transfer_learning_model(num_classes=num_classes)
    model.summary()

    callbacks = get_callbacks()

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = val_generator.samples // val_generator.batch_size
    if train_generator.samples % train_generator.batch_size != 0:
        steps_per_epoch += 1
    if val_generator.samples % val_generator.batch_size != 0:
        validation_steps += 1

    print(f'\nStarting training for {EPOCHS} epochs...')
    print(f'Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}')

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    np.save(os.path.join(os.path.dirname(MODEL_PATH), 'training_history.npy'), history.history)
    plot_training_history(history)

    best_epoch = np.argmax(history.history['val_accuracy'])
    final_val_acc = history.history['val_accuracy'][best_epoch]
    final_val_loss = history.history['val_loss'][best_epoch]

    print(f'\nTraining Complete!')
    print(f'Best epoch: {best_epoch + 1}')
    print(f'Best Validation Accuracy: {final_val_acc:.4f}')
    print(f'Best Validation Loss: {final_val_loss:.4f}')
    print(f'Model saved to: {MODEL_PATH}')

if __name__ == '__main__':
    main()
