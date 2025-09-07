# IMAGE-TRANSFORMATIONS
## NAME:GAYATHRI K
## REG NO:212223230061
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
<br>

### Step2:
<br>

### Step3:
<br>

### Step4:
<br>

### Step5:
<br>

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
<img width="623" height="470" alt="image" src="https://github.com/user-attachments/assets/6a724880-5d06-4817-993f-23f8b6357400" />

### i)Image Translation
<br>
<img width="632" height="470" alt="image" src="https://github.com/user-attachments/assets/da2e17fc-c463-46bd-bd69-31d3c323c6f2" />

<br>
<br>
<br>

### ii) Image Scaling
<br>
<img width="651" height="221" alt="image" src="https://github.com/user-attachments/assets/686cda70-af70-491a-ae2c-7c6a0598fce2" />


<br>
<br>
<br>


### iii)Image shearing
<br>
<img width="645" height="478" alt="image" src="https://github.com/user-attachments/assets/9cf3223b-ae78-4eb7-819f-1d20ef19d4fc" />


<br>
<br>
<br>


### iv)Image Reflection
<br>
<img width="646" height="476" alt="image" src="https://github.com/user-attachments/assets/a090de24-f63c-44e7-8e50-2a7544bde4ec" />

<br>
<br>
<br>



### v)Image Rotation
<br>
<img width="650" height="471" alt="image" src="https://github.com/user-attachments/assets/25d7e0a6-178e-4fc9-9cce-4f15ea1b43c6" />

<br>
<br>
<br>



### vi)Image Cropping
<br>
<img width="735" height="511" alt="image" src="https://github.com/user-attachments/assets/4d8efd72-87a3-48c7-b6cd-c9e5fa9ba1fe" />

<br>
<br>
<br>


## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
