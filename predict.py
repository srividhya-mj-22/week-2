import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
model = load_model("plant_disease_model.h5")

# Paths
test_folder = "dataset/test"
train_folder = "dataset/train"

# Class names
class_names = sorted(os.listdir(train_folder))

# Lists to store true and predicted labels
y_true = []
y_pred = []
image_names = []

# Limit number of images to display
max_display = 15
display_count = 0
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
axes = axes.flatten()

# Loop through test dataset
for cls in os.listdir(test_folder):
    cls_path = os.path.join(test_folder, cls)
    if not os.path.isdir(cls_path):
        continue
    for img_file in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_file)
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred = model.predict(img_array)
        class_idx = np.argmax(pred, axis=1)[0]
        
        # Store for metrics
        y_true.append(class_names.index(cls))
        y_pred.append(class_idx)
        image_names.append(img_file)
        
        # Display a limited number of images
        if display_count < max_display:
            axes[display_count].imshow(img)
            axes[display_count].set_title(f"Predicted: {class_names[class_idx]}")
            axes[display_count].axis('off')
            display_count += 1

plt.tight_layout()
plt.show()

# Classification report
print("✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
print("✅ Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save predictions to CSV
results = pd.DataFrame({
    "Image": image_names,
    "True_Class": [class_names[i] for i in y_true],
    "Predicted_Class": [class_names[i] for i in y_pred]
})
results.to_csv("test_predictions.csv", index=False)
print("✅ Predictions saved to test_predictions.csv")
