import numpy as np
from omnicloudmask import predict_from_array

filepath = "/home/dario/Desktop/sample_cloud_detection_data/NO_CLOUD_NPY/greece_tile_(0, 7424).npy"
# Load the image data   
img = np.load(filepath)
print(img.shape)

# change the shape to (3, height, width) for prediction
img = np.transpose(img, (2, 0, 1))

# Normalize to 0-1 range for model input
img = (img - np.min(img)) / (np.max(img) - np.min(img))
img = img.astype(np.float32)  # Use float32 for model input

print(img.shape)

img_rgnir = img[[3, 2, 4], :, :]  # Extract Red, Green, NIR bands

# Predict cloud and cloud shadow masks
pred_mask = predict_from_array(img_rgnir)

print(pred_mask.shape)

# Visualize the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Input Image")
# plot rgb image
plt.imshow(img.transpose(1, 2, 0)[:, :, [3, 2, 1]])  # RGB order
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Cloud Mask")
plt.imshow(pred_mask[0], cmap="gray")
plt.colorbar()
plt.axis("off")

plt.show()