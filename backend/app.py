import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import os
import io
from flask_cors import CORS


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTModel().to(device)
model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location=device))

model.eval()

# Flask app
app = Flask(__name__)
CORS(app)

# Fashion MNIST class labels
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 1 channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  # invert grayscale for Fashion MNIST
    transforms.Normalize((0.5,), (0.5,))  # normalize
])


def preprocess_img(file):
    img = Image.open(io.BytesIO(file.read())).convert("L")  # Convert to grayscale
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return img


def get_probabilities(output, top=3):
    probs = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
    probs = probs.squeeze().cpu().numpy()  # Convert tensor to NumPy array
    sorted_indices = probs.argsort()[::-1]
    sorted_probs = probs[sorted_indices][:top]
    sorted_probs = [f"{100.0 * p:.2f}%" for p in sorted_probs]
    sorted_labels = [class_labels[i] for i in sorted_indices][:top]

    return sorted_labels, sorted_probs


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = preprocess_img(file)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = class_labels[predicted.item()]
    labs, probs = get_probabilities(output, 3)

    payload = {
        "prediction": predicted_class,
        "labels": labs,
        "probabilities": probs,
    }
    return jsonify(payload)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)