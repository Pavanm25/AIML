import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define UNet model architecture
def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = conv4
    
    up5 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    
    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=conv8)
    
    return model

# Load and preprocess the brain CT image
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image from path:", image_path)
        return None
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    # Resize image to desired input size
    img = cv2.resize(img, (256, 256))
    # Expand dimensions to make it suitable for the UNet model
    img = np.expand_dims(img, axis=-1)
    return img

# Example usage
if __name__ == "__main__":
    # Define input image path
    input_image_path = r'D:\jain college folder\2nd Sem\ml datasets\archive (2)\brain_tumor_dataset\yes\no 923.jpg'
    
    # Preprocess input image
    input_image = preprocess_image(input_image_path)
    if input_image is None:
        exit()  # Exit if image loading failed
    
    # Build UNet model
    model = unet()
    
    # Compile the model
    # Compile the model with the correct optimizer initialization
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy')

    # Perform segmentation
    segmented_image = model.predict(np.expand_dims(input_image, axis=0))
    
    threshold = 0.8
    binary_mask = (segmented_image > threshold).astype(np.uint8)
    
    # Load original image
    original_image = cv2.imread(input_image_path)
    
    # Plot both original and segmented images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image[0, :, :, 0], cmap='gray')
    plt.title('Segmented Image')
    plt.axis('off')
    
    plt.show()
