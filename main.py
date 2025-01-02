
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

# Fetch the image from the URL
# url ="images/image.png"

url = 'https://media.geeksforgeeks.org/wp-content/uploads/20210924192026/bitss.jpg'
response = requests.get(url, stream=True).content
np_image = np.asarray(bytearray(response), dtype="uint8")
image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (11, 11), 0)

# Detect edges using Canny
canny = cv2.Canny(blur, 30, 150, 3)

# Dilate the edges to close gaps
dilated = cv2.dilate(canny, (1, 1), iterations=0)

# Find contours
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw contours on the original image
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)


# Print the number of objects detected
print("Object in the image:", len(contours))

# Display the processed image
plt.imshow(rgb)
plt.axis('off')
plt.title("Detected Objects")
plt.show()

