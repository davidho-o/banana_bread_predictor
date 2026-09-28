import tensorflow as tf

#loading the model
model = tf.keras.models.load_model('banana_model.keras')

image_path = 'data/test/ripe/musa-acuminata-banana-ad772802-394a-11ec-996c-d8c4975e38aa_jpg.rf.ac1e17757ffe79f22868f68252821f9d.jpg'

#loading the image cropped
img = tf.keras.utils.load_img(image_path, target_size = (256,256))

#the image is trasnformed t an array so it can be understood by the model
img_array = tf.keras.utils.img_to_array(img)

#the model is used to batches so we have to have the first number the number of photos we send so we add a 'column'
img_array = tf.expand_dims(img_array,axis=0)

#finally predicting
result = model.predict(img_array)

#it's saved as [[0.05, 0.85, 0.08, 0.02]] so we have to extract the biggest probability
winning_index = tf.argmax(result[0])

categories = ['overripe', 'ripe', 'rotten', 'unripe']

print(categories[winning_index])
