import os 
import cv2
import numpy as np
def dilate_images(image_dir, dilate_dir):
    """
    Dilate the images in the given directory and save the results.
    """
    for image_name in os.listdir(image_dir):
        if not image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        # 1. Read the image
        image = cv2.imread(os.path.join(image_dir, image_name))  # Replace with your image path
        if image is None:
            print("Unable to load the image. Please check the path!")
            exit()

        # 2. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. Dilate the image
        dilate_image = cv2.dilate(gray, kernel=np.ones((5, 5), np.uint8), iterations=1)
        cv2.imwrite(os.path.join(dilate_dir, f"dilate_{image_name}"), dilate_image)
        base_name = os.path.splitext(image_name)[0]
        with open(f"annotations/{base_name}.txt", "r") as f1, open(f"annotations/dilate_{base_name}.txt", "w") as f2:
            for line in f1.readlines():
                f2.write(line)

if __name__ == "__main__":
    base_dir = "data/square/base_data"
    dilate_dir = "data/square/dilate_data"
    os.makedirs(dilate_dir, exist_ok=True)
    dilate_images(base_dir, dilate_dir)

    # base_dir = "data/rectangle/base_data"
    # dilate_dir = "data/rectangle/dilate_data"
    # os.makedirs(dilate_dir, exist_ok=True)
    # dilate_images(base_dir, dilate_dir)

    # base_dir = "data/strip/base_data"
    # dilate_dir = "data/strip/dilate_data"
    # os.makedirs(dilate_dir, exist_ok=True)
    # dilate_images(base_dir, dilate_dir)