import tensorflow as tf

#for the training set
train_dir_path = 'data/train'
train_data = tf.keras.utils.image_dataset_from_directory(train_dir_path)

#for the validation set
valid_dir_path = 'data/valid'
valid_data = tf.keras.utils.image_dataset_from_directory(valid_dir_path)

#base model
base_model = tf.keras.applications.MobileNetV2( #famous, created by Google
    input_shape=(256, 256, 3),  # how the photos will be percieved
    include_top=False, # the top layer that will be replaced by our banana one
    weights='imagenet' #all the rules that the model already knows
)
base_model.trainable = False #so that it does not fuck up the base model and retrain himself incorrectly

#the model we are going to work with
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), 
    tf.keras.layers.Dense(4, activation='softmax') # transforming the output in predictions, each cumulating to 100%
])

model.compile(
    optimizer='adam', # method used by the model to learn and correct mistakes
    loss='sparse_categorical_crossentropy', # the way it calculates how bad he was mistaken
    metrics=['accuracy'] # the accuracy of the answer
)

training_history = model.fit(
    train_data,
    validation_data = valid_data,
    epochs = 8
)

model.save('banana_model.keras')
