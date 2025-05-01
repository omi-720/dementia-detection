import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from glob import glob
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMG_SIZE = 128  # Size to resize images
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 4  # mild, moderate, non-demented, very mild
LEARNING_RATE = 0.001

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths for dataset
ORIGINAL_DATA_DIR = 'dementia_dataset'  # Directory with MildDemented, ModerateDemented, etc. folders
PROCESSED_DATA_DIR = 'processed_dataset'  # Directory to store the split data

# Create processed data directory structure
def create_directory_structure():
    # Create main directories
    for split in ['train', 'val', 'test']:
        for class_name in ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']:
            os.makedirs(os.path.join(PROCESSED_DATA_DIR, split, class_name), exist_ok=True)
    
    print("Created directory structure for processed dataset.")

# Split the dataset into train, validation, and test sets
def split_dataset():
    # Check if the split has already been done
    if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'split_done.txt')):
        print("Dataset already split. Skipping split process.")
        return
    
    print("Splitting dataset into train, validation, and test sets...")
    
    # Process each class directory
    for class_name in ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']:
        # Get all image paths for this class
        class_path = os.path.join(ORIGINAL_DATA_DIR, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist. Skipping.")
            continue
            
        image_paths = glob(os.path.join(class_path, '*.jpg'))
        image_paths.extend(glob(os.path.join(class_path, '*.jpeg')))
        image_paths.extend(glob(os.path.join(class_path, '*.png')))
        
        if len(image_paths) == 0:
            print(f"Warning: No images found in {class_path}. Skipping.")
            continue
            
        print(f"Found {len(image_paths)} images for {class_name}")
        
        # Split paths into train, validation, and test
        train_paths, temp_paths = train_test_split(
            image_paths, train_size=TRAIN_RATIO, random_state=42
        )
        
        # Further split temp_paths into validation and test
        relative_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_paths, test_paths = train_test_split(
            temp_paths, train_size=relative_ratio, random_state=42
        )
        
        print(f"{class_name} split: {len(train_paths)} train, {len(val_paths)} validation, {len(test_paths)} test")
        
        # Copy files to their respective directories
        for paths, split_name in [(train_paths, 'train'), (val_paths, 'val'), (test_paths, 'test')]:
            dest_dir = os.path.join(PROCESSED_DATA_DIR, split_name, class_name)
            
            for src_path in tqdm(paths, desc=f"Copying {split_name} {class_name}"):
                filename = os.path.basename(src_path)
                dst_path = os.path.join(dest_dir, filename)
                shutil.copy(src_path, dst_path)
    
    # Create a marker file to indicate split is done
    with open(os.path.join(PROCESSED_DATA_DIR, 'split_done.txt'), 'w') as f:
        f.write('Dataset split completed')
    
    print("Dataset split and organization completed.")

# Set up paths for the processed dataset
def setup_data_paths():
    # If the dataset hasn't been processed yet, do it
    if not os.path.exists(PROCESSED_DATA_DIR):
        create_directory_structure()
        split_dataset()
    
    # Define the paths
    TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
    VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')
    TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')
    
    return TRAIN_DIR, VAL_DIR, TEST_DIR

# Setup data paths
TRAIN_DIR, VAL_DIR, TEST_DIR = setup_data_paths()

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Just rescaling for validation and test sets
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class indices to understand model output later
class_indices = train_generator.class_indices
print("Class indices:", class_indices)
class_names = list(class_indices.keys())

# Define CNN model architecture
def create_model():
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# Create and compile the model
model = create_model()
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Define callbacks for training
checkpoint = ModelCheckpoint(
    'dementia_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Train the model
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('dementia_classification_model.h5')
print("Model saved as 'dementia_classification_model.h5'")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=max(1, test_generator.samples // BATCH_SIZE))
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Generate predictions on test data
test_generator.reset()
y_pred = model.predict(test_generator, steps=max(1, test_generator.samples // BATCH_SIZE))
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = test_generator.classes[:len(y_pred_classes)]

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Function to create TFLite version if needed
def convert_to_tflite():
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('dementia_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'dementia_model.tflite'")

# Uncomment if you need TFLite format
# convert_to_tflite()