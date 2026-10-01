import json
import tensorflow as tf

# same folder used in train.py
train_data = tf.keras.utils.image_dataset_from_directory('data/train')

class_names = train_data.class_names   # folder names in alphabetical order
print(class_names)

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)          # the app reads this file