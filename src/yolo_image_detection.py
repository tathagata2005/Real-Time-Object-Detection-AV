from ultralytics import YOLO
import cv2
import os
import glob

# ==========================================
# 1. Load YOLO model
# ==========================================

model = YOLO("yolo11n.pt")

# ==========================================
# 2. Input and output folders
# ==========================================

input_folder = "datasets/test_images"
output_folder = "outputs/yolo_results"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 3. Find all images
# ==========================================

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.avif", "*.tiff", "*.webp"]

image_paths = []

for extension in image_extensions:
    image_paths.extend(
        glob.glob(os.path.join(input_folder, extension))
    )

print(f"\nFound {len(image_paths)} images.")

# ==========================================
# 4. Check if images exist
# ==========================================

if len(image_paths) == 0:
    print("No images found!")
    print("Put your images inside:")
    print(input_folder)
    exit()

# ==========================================
# 5. Process every image
# ==========================================

for image_path in image_paths:

    print("\n" + "=" * 50)
    print("Processing:", image_path)

    # Run YOLO
    results = model(image_path, verbose=False)

    result = results[0]

    # --------------------------------------
    # Print detected objects
    # --------------------------------------

    print("Detected Objects:")

    if len(result.boxes) == 0:
        print("No objects detected.")

    else:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                f"  {class_name} "
                f"(Confidence: {confidence:.2f})"
            )

    # --------------------------------------
    # Draw bounding boxes
    # --------------------------------------

    annotated_image = result.plot()

    # --------------------------------------
    # Create output filename
    # --------------------------------------

    filename = os.path.basename(image_path)

    output_path = os.path.join(
        output_folder,
        "detected_" + filename
    )

    # Save image
    cv2.imwrite(output_path, annotated_image)

    print("Saved:", output_path)

# ==========================================
# 6. Finished
# ==========================================

print("\n" + "=" * 50)
print("ALL IMAGES PROCESSED SUCCESSFULLY!")
print("=" * 50)

print("\nResults are available in:")
print(output_folder)