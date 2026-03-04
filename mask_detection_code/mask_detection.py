import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path=r'C:\Users\alkes\OneDrive\Desktop\mask detection' 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

train_generator=train_datagen.flow_from_directory(
    dataset_path +"\\training",
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator=train_datagen.flow_from_directory(
    dataset_path + "\\training",
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'   
)

test_datagen=ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(
    dataset_path + "\\testing",
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
     Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
     Conv2D(512,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(512,activation='relu'),
    Dropout(0.5),
    Dense(2,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_generator,validation_data=val_generator,epochs=20)

img_path=r'C:\Users\alkes\OneDrive\Desktop\mask detection\testing\with_mask\110-with-mask.jpg'

class_labels=['with_mask','without_mask']

img = load_img(img_path, target_size=(150,150))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction=model.predict(img_array)
predicted_index=np.argmax(prediction, axis=1)[0]
predicted_label=class_labels[predicted_index]
confidence=np.max(prediction) *100

print(f'\npredicted class:{predicted_label}')
print(f'confidence:{confidence:.2f}%')

plt.imshow(img)
plt.title(f'prediction:{predicted_label} ({confidence:.2f}%)')
plt.axis('off')

plt.savefig('prediction_result.png')
print("Prediction image saved as prediction_result.png")