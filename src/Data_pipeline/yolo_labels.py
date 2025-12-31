import os
from PIL import Image, ImageDraw, ImageFont

# Standard 12-lead ECG order
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def generate_yolo_labels(image_path, output_dir):
    """
    Generates a YOLO format label file for a given GenECG image.

    The function divides the image into a 3x4 grid, creating 12 bounding boxes
    for the standard 12 ECG leads. A buffer is included to ensure wave peaks
    are not cut off.

    Args:
        image_path (str): The path to the input ECG image.
        output_dir (str): The directory to save the .txt label file.
    """
    try:
        # Load the image to get its dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get the base name of the image file to create the corresponding .txt file
        base_name = os.path.basename(image_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_dir, f"{file_name_no_ext}.txt")

        # Grid dimensions
        rows, cols = 3, 4
        cell_width = img_width / cols
        cell_height = img_height / rows

        # Buffer calculation (e.g., 5% of cell dimensions)
        buffer_x = cell_width * 0.05
        buffer_y = cell_height * 0.05

        yolo_labels = []

        # Iterate through columns then rows to match the specified lead order
        for col in range(cols):
            for row in range(rows):
                # Calculate class_id based on column-major order
                class_id = col * rows + row

                # Calculate bounding box with buffer
                x_min = col * cell_width + buffer_x
                y_min = row * cell_height + buffer_y
                box_width = cell_width - (2 * buffer_x)
                box_height = cell_height - (2 * buffer_y)

                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x_min + box_width / 2) / img_width
                y_center = (y_min + box_height / 2) / img_height
                norm_width = box_width / img_width
                norm_height = box_height / img_height
                
                # Append to list, ensuring values are within [0, 1]
                yolo_labels.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                )

        # Save the .txt file
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_labels))
        
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
    except Exception as e:
        print(f"An error occurred with image {image_path}: {e}")
    else:
        # Only print success if no exception was raised
        print(f"Successfully generated YOLO label file: {output_path}")

def visualize_detections(image, boxes):
    """
    Draws bounding boxes and labels on an image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        boxes (list): A list of strings, where each string is a YOLO-formatted
                      bounding box: "class_id x_center y_center width height".
    
    Returns:
        PIL.Image.Image: The image with detections drawn on it.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    for box_str in boxes:
        parts = box_str.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        norm_width = float(parts[3])
        norm_height = float(parts[4])

        # Denormalize coordinates
        box_width = norm_width * img_width
        box_height = norm_height * img_height
        x_min = (x_center * img_width) - (box_width / 2)
        y_min = (y_center * img_height) - (box_height / 2)
        x_max = x_min + box_width
        y_max = y_min + box_height

        # Draw bounding box
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
        
        # Get label and draw it
        label = LEAD_ORDER[class_id]
        # Offset the text slightly to be more readable
        draw.text((x_min + 5, y_min + 5), label, fill="red")

    return img


if __name__ == '__main__':
    # This is an example of how to use the functions.
    # It creates a dummy image, generates labels, and then visualizes them.
    
    print("Running example usage of generate_yolo_labels and visualize_detections...")
    
    # Setup paths
    dummy_dir = "dummy_data_visualize"
    dummy_labels_dir = os.path.join(dummy_dir, "labels")
    dummy_image_path = os.path.join(dummy_dir, "sample_ecg.png")
    labeled_image_path = os.path.join(dummy_dir, "sample_ecg_labeled.png")

    os.makedirs(dummy_labels_dir, exist_ok=True)

    try:
        # Create a blank 1024x768 image
        dummy_img = Image.new('RGB', (1024, 768), color='white')
        dummy_img.save(dummy_image_path)
        print(f"Created a dummy image at: {dummy_image_path}")
        
        # 1. Generate YOLO labels
        generate_yolo_labels(dummy_image_path, dummy_labels_dir)

        # 2. Visualize the detections
        label_path = os.path.join(dummy_labels_dir, "sample_ecg.txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                boxes = f.readlines()
            
            # Load the original image
            image_to_visualize = Image.open(dummy_image_path)
            
            # Draw detections
            labeled_image = visualize_detections(image_to_visualize, boxes)
            
            # Save the result
            labeled_image.save(labeled_image_path)
            print(f"\nSuccessfully generated and saved labeled image to: {labeled_image_path}")

    except Exception as e:
        print(f"An error occurred during the example run: {e}")
