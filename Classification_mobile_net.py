from keras.applications import MobileNet
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# MobileNet works with images of dimension 224x224
img_rows, img_cols = 224, 224

# Load MobileNet model with ImageNet weights, exclude the top layers
MobileNet_base = MobileNet(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Unfreeze all layers of MobileNet for training
for layer in MobileNet_base.layers:
    layer.trainable = True

# Function to add a new head (top) to MobileNet
def addTopModelMobileNet(bottom_model, num_classes):
    """Creates the top layers that will be added on top of the base MobileNet model"""
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    
    return top_model

# Number of output classes (for example, 5 classes in the dataset)
num_classes = 5

# Add the custom head to the MobileNet model
FC_Head = addTopModelMobileNet(MobileNet_base, num_classes)

# Create the full model by combining base MobileNet and the custom top layers
model = Model(inputs=MobileNet_base.input, outputs=FC_Head)

# Print model summary
print(model.summary())

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,
    width_shift_range=0.5,
    height_shift_range=0.5,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Path to training and validation datasets
train_data_dir = '/path/to/train/directory'
validation_data_dir = '/path/to/validation/directory'

# Batch size
batch_size = 32

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model checkpoints and early stopping
checkpoint = ModelCheckpoint(
    'emotion_face_mobilNet.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=5,
    verbose=1,
    factor=0.2,
    min_lr=0.0001
)

# Callbacks list
callbacks = [earlystop, checkpoint, learning_rate_reduction]

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

# Number of training and validation samples
nb_train_samples = 24177
nb_validation_samples = 3005

# Number of training epochs
epochs = 25

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)
