import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import the model architecture
from mobilenet_lstm_model import build_mobilenet_lstm_model


def load_preprocessed_data(data_dir='processed_data'):
    """
    Load preprocessed .npy files from the base repo
    Assumes structure: processed_data/word_name/take_X.npy
    """
    X = []
    y = []
    word_labels = []
    
    # Iterate through word directories
    for word_idx, word_name in enumerate(sorted(os.listdir(data_dir))):
        word_path = os.path.join(data_dir, word_name)
        
        if not os.path.isdir(word_path):
            continue
            
        word_labels.append(word_name)
        print(f"Loading word: {word_name}")
        
        # Load all takes for this word
        take_count = 0
        for file_name in os.listdir(word_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(word_path, file_name)
                frames = np.load(file_path)
                
                # Ensure correct shape (22, 80, 112)
                if frames.shape == (22, 80, 112):
                    X.append(frames)
                    y.append(word_idx)
                    take_count += 1
                else:
                    print(f"  Warning: Skipping {file_name} - incorrect shape {frames.shape}")
        
        print(f"  Loaded {take_count} takes for '{word_name}'")
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension: (samples, 22, 80, 112) -> (samples, 22, 80, 112, 1)
    X = np.expand_dims(X, axis=-1)
    
    # Normalize to [0, 1] if not already
    if X.max() > 1.0:
        X = X / 255.0
    
    print(f"\n✅ Loaded {len(X)} samples across {len(word_labels)} classes")
    print(f"Data shape: {X.shape}")
    print(f"Word labels: {word_labels}")
    
    return X, y, word_labels


def augment_batch(X_batch, y_batch):
    """
    Data augmentation for video sequences
    """
    augmented_X = []
    augmented_y = []
    
    for X, y in zip(X_batch, y_batch):
        # Original
        augmented_X.append(X)
        augmented_y.append(y)
        
        # Horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            X_flipped = np.flip(X, axis=2)  # Flip width dimension
            augmented_X.append(X_flipped)
            augmented_y.append(y)
        
        # Brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            X_bright = np.clip(X * brightness_factor, 0, 1)
            augmented_X.append(X_bright)
            augmented_y.append(y)
        
        # Random temporal shifts (shift frames slightly)
        if np.random.rand() > 0.5:
            shift = np.random.randint(-2, 3)
            X_shifted = np.roll(X, shift, axis=0)
            augmented_X.append(X_shifted)
            augmented_y.append(y)
    
    return np.array(augmented_X), np.array(augmented_y)


def train_model(data_dir='processed_data', epochs=50, batch_size=16):
    """
    Main training function
    """
    
    # Load data
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    X, y, word_labels = load_preprocessed_data(data_dir)
    
    num_classes = len(word_labels)
    
    # Split data
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MOBILENET-LSTM MODEL")
    print("="*60)
    model = build_mobilenet_lstm_model(
        num_frames=22,
        num_classes=num_classes,
        img_height=80,
        img_width=112
    )
    
    # Print model summary
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compile model with label smoothing
    print("\n" + "="*60)
    print("COMPILING MODEL")
    print("="*60)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
    )
    print("✅ Model compiled successfully")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'model/best_mobilenet_lstm.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir='../logs',
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    history = model.fit(
        X_train, y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('model/mobilenet_lstm_final.h5')
    print("\n✅ Final model saved as 'mobilenet_lstm_final.h5'")
    
    # Save word labels
    np.save('model/word_labels.npy', word_labels)
    print("✅ Word labels saved as 'word_labels.npy'")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    val_loss, val_acc, val_top3 = model.evaluate(X_val, y_val_cat)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Top-3 Accuracy: {val_top3*100:.2f}%")
    
    return model, history, word_labels


def plot_training_history(history):
    """
    Plot training curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_mobilenet_lstm.png')
    print("✅ Training history plot saved as 'training_history_mobilenet_lstm.png'")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MOBILENET-LSTM LIP READER TRAINING")
    print("="*60)
    
    # Check if processed_data directory exists (try both locations)
    if os.path.exists('../processed_data'):
        data_dir = '../processed_data'
    elif os.path.exists('processed_data'):
        data_dir = 'processed_data'
    else:
        print("\n❌ ERROR: 'processed_data' directory not found!")
        print("Please run preprocess.py first to create processed data.")
        exit(1)
    
    print(f"Using data directory: {data_dir}")
    
    # Start training
    try:
        model, history, word_labels = train_model(
            data_dir=data_dir,
            epochs=50,
            batch_size=16
        )
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("✅ Model saved as: best_mobilenet_lstm.h5")
        print("✅ Word labels saved as: word_labels.npy")
        print("✅ Training plot saved as: training_history.png")
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()