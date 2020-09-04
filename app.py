from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import torch
import urllib.request
from PIL import Image
from fire_classifier import model

model.classifier.load_state_dict(torch.load('fire_classifier'))
model.eval()

def process_image(image_path):
    image_path = urllib.request.urlopen(image_path)
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size 
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    img = np.array(img)
    
    img = img.transpose((2, 0, 1))
    
    img = img/255
    
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    img = img[np.newaxis,:]
    image = torch.from_numpy(img)
    image = image.float()
    return image[:, :3]

def predict(image, model):
    output = model.forward(image)
    output = torch.exp(output)
    
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

app = Flask(__name__)
CORS(app)

@app.route('/')
def test():
    return 'HotSpot API running!'

@app.route('/classify', methods=['POST'])
def classify():
    try:
        image_url = request.get_json()["url"]
        if image_url is None or len(image_url) == 0:
            return jsonify({'error': {'msg': 'No image URL or empty image URL provided. Please provide a valid image URL.'}})
        confidence, classification = predict(process_image(image_url), model)
        classification = 'fire' if classification == 0 else 'no fire'
        return jsonify({'confidence': confidence, 'classification': classification, 'error': None})
    except:
        return jsonify({'error': {'msg': 'Unable to classify image. Please make sure a valid image URL is provided.'}})

