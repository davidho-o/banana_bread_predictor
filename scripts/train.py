import tensorflow as tf

#for the training set
train_dir_path = 'data/train'
train_data = tf.keras.utils.image_dataset_from_directory(train_dir_path)

#for the validation set
valid_dir_path = 'data/valid'
valid_data = tf.keras.utils.image_dataset_from_directory(valid_dir_path)
