from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

# Model ve işlemciyi yükleme
trained_model_path = "./trained_model"
model_name_Blip = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name_Blip)
model_Blip = BlipForConditionalGeneration.from_pretrained(trained_model_path)

# Cihaz ayarları
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_Blip = model_Blip.to(device)

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = "redshirt.jpg" #request.files['image']
    
    try:
        # Resim dosyasını aç
        image = Image.open(image_file)
        
        # Resmi işleme
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model_Blip.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        print("caption", caption)

        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)