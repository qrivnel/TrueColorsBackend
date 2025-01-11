import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image, ExifTags
import base64
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/color-blindness', methods=['POST'])
def color_blindness():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    image = fix_image_orientation(image)
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.ndim == 3 and image_np.shape[2] == 3:  # RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    processed_images = []
    for typeOfBlindness in ["Protan", "Deutan", "Tritan"]:
        processed_image = np.copy(image_np)
        processed_image = for_color_blindnesses(processed_image, typeOfBlindness)

        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        processed_images.append(processed_image_base64)

    # save_processed_images(processed_images)

    return jsonify(processed_images)

PROTANOPIA_MATRIX = np.array([
    [0.567, 0.433, 0.0],
    [0.558, 0.442, 0.0],
    [0.0,   0.242, 0.758]
])

DEUTERANOPIA_MATRIX = np.array([
    [0.625, 0.375, 0.0],
    [0.7,   0.3,   0.0],
    [0.0,   0.3,   0.7]
])

TRITANOPIA_MATRIX = np.array([
    [0.95,  0.05,  0.0],
    [0.0,   0.433, 0.567],
    [0.0,   0.475, 0.525]
])

@app.route('/simulation', methods=['POST'])
def simulation():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    image = fix_image_orientation(image)
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.ndim == 3 and image_np.shape[2] == 3:  # RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    processed_images = []
    for typeOfBlindness in ["Protan", "Deutan", "Tritan"]:
        processed_image = np.copy(image_np)
        processed_image = simulate_color_blindness2(processed_image, typeOfBlindness)

        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        processed_images.append(processed_image_base64)

    # save_processed_images(processed_images)

    return jsonify(processed_images)

def simulate_color_blindness(image_data, type_of_blindness):
    if type_of_blindness == "Protan":
        transformation_matrix = PROTANOPIA_MATRIX
    if type_of_blindness == "Deutan":
        transformation_matrix = DEUTERANOPIA_MATRIX
    if type_of_blindness == "Tritan":
        transformation_matrix = TRITANOPIA_MATRIX
    height, width, _ = image_data.shape
    new_image_data = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            original_color = image_data[y, x, :3] / 255.0
            transformed_color = np.dot(transformation_matrix, original_color)
            new_image_data[y, x, :3] = np.clip(transformed_color * 255, 0, 255)
    return new_image_data

def simulate_color_blindness2(image_data, type_of_blindness):
    if type_of_blindness == "Protan":
        transformation_matrix = PROTANOPIA_MATRIX
    elif type_of_blindness == "Deutan":
        transformation_matrix = DEUTERANOPIA_MATRIX
    elif type_of_blindness == "Tritan":
        transformation_matrix = TRITANOPIA_MATRIX
    else:
        raise ValueError("Invalid type of blindness")

    normalized_image = image_data[:, :, :3] / 255.0

    transformed_image = np.dot(normalized_image.reshape(-1, 3), transformation_matrix.T)

    transformed_image = np.clip(transformed_image, 0, 1).reshape(image_data.shape[0], image_data.shape[1], 3)
    transformed_image = (transformed_image * 255).astype(np.uint8)

    return transformed_image


def for_color_blindnesses(image, type_of_blindness):
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            blue = image[y, x][0]
            green = image[y, x][1]
            red = image[y, x][2]

            if type_of_blindness == "Protan":
                if red <= 190:
                    image[y, x][2] += 65
                else:
                    image[y, x][2] = 255
                if blue >= 30:
                    image[y, x][0] -= 30
                else:
                    image[y, x][0] = 0
                if green >= 30:
                    image[y, x][1] -= 30
                else:
                    image[y, x][1] = 0

            elif type_of_blindness == "Deutan":
                if green <= 225:
                    image[y, x][1] += 30
                else:
                    image[y, x][1] = 255
                if blue >= 30:
                    image[y, x][0] -= 30
                else:
                    image[y, x][0] = 0
                if red <= 235:
                    image[y, x][2] += 20
                else:
                    image[y, x][2] = 255

            elif type_of_blindness == "Tritan":
                if blue <= 225:
                    image[y, x][0] += 30
                else:
                    image[y, x][0] = 255
                if green >= 30:
                    image[y, x][1] -= 30
                else:
                    image[y, x][1] = 0
                if red >= 20:
                    image[y, x][2] -= 20
                else:
                    image[y, x][2] = 0
    return image


def for_color_blindnesses_v2(image, type_of_blindness):
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            blue = image[y, x][0]
            green = image[y, x][1]
            red = image[y, x][2]

            if type_of_blindness == "Protan" and red > 100:
                if red <= 190:
                    image[y, x][2] = min(red + 65, 255)
                else:
                    image[y, x][2] = 255
                image[y, x][0] = max(blue - 30, 0)
                image[y, x][1] = max(green - 30, 0)

            elif type_of_blindness == "Deutan" and green > 100 and red > 100:
                image[y, x][1] = min(green + 30, 255)
                image[y, x][0] = max(blue - 30, 0)
                image[y, x][2] = min(red + 20, 255)

            elif type_of_blindness == "Tritan" and blue > 100:
                image[y, x][0] = min(blue + 30, 255)
                image[y, x][1] = max(green - 30, 0)
                image[y, x][2] = max(red - 20, 0)
    return image


def fix_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None and orientation in exif:
            orientation_value = exif[orientation]

            # EXIF yön bilgisine göre döndürme
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Error fixing orientation: {e}")
    return image


def save_processed_images(processed_images, output_folder="output_images"):
    import os
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, image_base64 in enumerate(processed_images):
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Görüntüyü kaydet
        output_path = os.path.join(output_folder, f"processed_image_{idx+1}.jpg")
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)