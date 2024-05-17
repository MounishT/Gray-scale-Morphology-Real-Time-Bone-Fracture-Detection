# Gray-scale-Morphology-Real-Time-Bone-Fracture-Detection
## Aim:
To represent the bone fracture detection using Gray Scale Morphology.

### DEVELOPED BY:T MOUNISH
### REGISTER NUMBER : 212223240098
## CODE:
```
import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_fractures(image):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edges = cv2.Canny(dilation, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image

def present_results(original_image, processed_image):
    # Display original and processed images
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Fracture Detected Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'bone.jpeg'
image = cv2.imread(image_path)

processed_image = preprocess_image(image)
fracture_detected_image = detect_fractures(processed_image)

present_results(image, fracture_detected_image)

```

## OUTPUT:
![image](https://github.com/saiganesh2006/Gray-scale-Morphology-Real-Time-Bone-Fracture-Detection/assets/145742342/41242df1-22f2-41a5-a841-5a8dbc742d9f)
![image](https://github.com/saiganesh2006/Gray-scale-Morphology-Real-Time-Bone-Fracture-Detection/assets/145742342/2cb7de89-124e-4df4-87d1-7cbd27d3a9da)

## The advantages and challenges of using morphological operations for this specific medical application:

### Advantages:
1.Robustness in handling noise and variations in image quality, crucial for medical images prone to artifacts.

2.Computational efficiency, allowing real-time processing feasible even with limited computational resources.

3.Interpretability, as morphological operations often preserve the structure and spatial relationships in the images, aiding in diagnosis.

### Challenges:
1.Parameter tuning, as selecting appropriate parameters for morphological operations can be challenging and may require expertise.

2.Sensitivity to image quality, particularly in cases of low-resolution or noisy images, where morphological operations might produce suboptimal results.

3.Limited ability to capture complex features, leading to potential false positives or negatives, especially in cases of intricate fractures or overlapping structures.

## RESULT:
Thus ,Successfully preprocesses an X-ray image, applies grayscale morphology for fracture detection, and displays the original and processed images, outlining potential fractures.





