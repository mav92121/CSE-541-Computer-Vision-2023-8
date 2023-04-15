import tensorflow as tf
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# Define the path to the saved model
# model_path = "myMode.h5"
# with open("model_trained.p", "rb") as f:
#     model = pickle.load(f)
#     import pickle


# Train and save a Keras model
# model = # train your model here
# model.save("my_model.h5")

# Load the saved model and save it using pickle
model = load_model("myMode.h5")
# with open("model_trained.p", "wb") as f:
    # pickle.dump(model, f)

# Load the saved model
# model = tf.keras.models.load_model(model_path)

# def getCalssName(classNo):
#     if   classNo == 0: return 'Speed Limit 20 km/h'
#     elif classNo == 1: return 'Speed Limit 30 km/h'
#     elif classNo == 2: return 'Speed Limit 50 km/h'
#     elif classNo == 3: return 'Speed Limit 60 km/h'
#     elif classNo == 4: return 'Speed Limit 70 km/h'
#     elif classNo == 5: return 'Speed Limit 80 km/h'
#     elif classNo == 6: return 'End of Speed Limit 80 km/h'
#     elif classNo == 7: return 'Speed Limit 100 km/h'
#     elif classNo == 8: return 'Speed Limit 120 km/h'
#     elif classNo == 9: return 'No passing'
#     elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
#     elif classNo == 11: return 'Right-of-way at the next intersection'
#     elif classNo == 12: return 'Priority road'
#     elif classNo == 13: return 'Yield'
#     elif classNo == 14: return 'Stop'
#     elif classNo == 15: return 'No vechiles'
#     elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
#     elif classNo == 17: return 'No entry'
#     elif classNo == 18: return 'General caution'
#     elif classNo == 19: return 'Dangerous curve to the left'
#     elif classNo == 20: return 'Dangerous curve to the right'
#     elif classNo == 21: return 'Double curve'
#     elif classNo == 22: return 'Bumpy road'
#     elif classNo == 23: return 'Slippery road'
#     elif classNo == 24: return 'Road narrows on the right'
#     elif classNo == 25: return 'Road work'
#     elif classNo == 26: return 'Traffic signals'
#     elif classNo == 27: return 'Pedestrians'
#     elif classNo == 28: return 'Children crossing'
#     elif classNo == 29: return 'Bicycles crossing'
#     elif classNo == 30: return 'Beware of ice/snow'
#     elif classNo == 31: return 'Wild animals crossing'
#     elif classNo == 32: return 'End of all speed and passing limits'
#     elif classNo == 33: return 'Turn right ahead'
#     elif classNo == 34: return 'Turn left ahead'
#     elif classNo == 35: return 'Ahead only'
#     elif classNo == 36: return 'Go straight or right'
#     elif classNo == 37: return 'Go straight or left'
#     elif classNo == 38: return 'Keep right'
#     elif classNo == 39: return 'Keep left'
#     elif classNo == 40: return 'Roundabout mandatory'
#     elif classNo == 41: return 'End of no passing'
#     elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

# Define the class labels
class_labels=[]
for i in range(0,43):
    class_labels.append(str(i))
# class_labels = ["class1", "class2", ..., "class42"]

# Define the path to the input image
image_path = "myData/4/4_6815_1577671995.765067.png"

# Load the input image and preprocess it
img = cv2.imread(image_path)


# Load the image and display it
# img = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Print the predicted class label
# print("Predicted class: ", pred_class)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (32, 32))
normalized = resized / 255.0
input_img = np.expand_dims(normalized, axis=-1)

pred_probs = model.predict(np.expand_dims(input_img, axis=0))[0]

# Get the predicted class label
pred_class = class_labels[np.argmax(pred_probs)]

# Print the predicted probabilities for each class
for i, prob in enumerate(pred_probs):
    print(f"{class_labels[i]}: {prob}")
# Predict the class of the input image
# pred = model.predict(np.expand_dims(input_img, axis=0))

# Get the predicted class label
# pred_class = class_labels[np.argmax(pred)]

# Print the predicted class label
print("Predicted class: ", pred_class)
