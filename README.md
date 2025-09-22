## EXP-3
# IMAGE-TRANSFORMATIONS
## NAME:GAYATHRI K
## REG NO:212223230061
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
## Step1:
Import necessary libraries such as OpenCV, NumPy, and Matplotlib for image processing and visualization.

## Step2:
Read the input image using cv2.imread() and store it in a variable for further processing.

## Step3:
Apply various transformations like translation, scaling, shearing, reflection, rotation, and cropping by defining corresponding functions:

1.Translation moves the image along the x or y-axis. 2.Scaling resizes the image by scaling factors. 3.Shearing distorts the image along one axis. 4.Reflection flips the image horizontally or vertically. 5.Rotation rotates the image by a given angle.

## Step4:
Display the transformed images using Matplotlib for visualization. Convert the BGR image to RGB format to ensure proper color representation.

## Step5:
Save or display the final transformed images for analysis and use plt.show() to display them inline in Jupyter or compatible environments.

## Program:
````
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rows, cols, _ = image.shape

M_translate = np.float32([[1, 0, 50], [0, 1, 100]])
translated_image = cv2.warpAffine(image_rgb, M_translate, (cols, rows))

scaled_image = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

M_shear = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
sheared_image = cv2.warpAffine(image_rgb, M_shear, (int(cols * 1.5), int(rows * 1.5)))

reflected_image = cv2.flip(image_rgb, 1)

M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(image_rgb, M_rotate, (cols, rows))

cropped_image = image_rgb[50:300, 100:400]

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(translated_image)
plt.title("Translated Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(scaled_image)
plt.title("Scaled Image")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(reflected_image)
plt.title("Reflected Image")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(rotated_image)
plt.title("Rotated Image")
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(cropped_image)
plt.title("Cropped Image")
plt.axis('off')
plt.show()



``````
## Output:
## ORIGINAL IMAGE
<img width="482" height="357" alt="image" src="https://github.com/user-attachments/assets/c40da483-1c18-4a32-bc93-ef342bc6db66" />


### i)Image Translation
<br>
<img width="483" height="350" alt="image" src="https://github.com/user-attachments/assets/234e8ecb-3954-49d0-b8a9-d0e410ba8ee4" />

<br>
<br>
<br>

### ii) Image Scaling
<br>
<img width="492" height="157" alt="image" src="https://github.com/user-attachments/assets/503ccbdb-4aff-4417-9e44-3d82b135d77f" />



<br>
<br>
<br>


### iii)Image shearing
<br>
<img width="478" height="341" alt="image" src="https://github.com/user-attachments/assets/6eec08be-27e7-45f2-bdf5-c0607f4da249" />


<br>
<br>
<br>


### iv)Image Reflection
<br>
<img width="479" height="342" alt="image" src="https://github.com/user-attachments/assets/7046cc01-f8ce-4216-b3a7-3474f5e24f1f" />


<br>
<br>
<br>



### v)Image Rotation
<br>
<img width="479" height="342" alt="image" src="https://github.com/user-attachments/assets/46d89398-969e-4acc-8b92-eef084cb348a" />


<br>
<br>
<br>



### vi)Image Cropping
<br>
<img width="478" height="383" alt="image" src="https://github.com/user-attachments/assets/a4d2de96-52c0-4ddf-bae7-b4d216467c93" />


<br>
<br>
<br>


## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
