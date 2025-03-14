import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog


# Define the same model architecture
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


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTModel().to(device)
model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location=device))
model.eval()

# Define class names (same as FashionMNIST dataset)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
print(class_names)


# Image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
        transforms.Resize((28, 28)),  # Resize to 28x28 (same as dataset)
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # Invert grayscale
        transforms.Normalize((0.5,), (0.5,))  # Normalize like training data
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


def get_probabilities(output, top=3):
    probs = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
    probs = probs.squeeze().cpu().numpy()  # Convert tensor to NumPy array
    sorted_indices = probs.argsort()[::-1]
    sorted_probs = probs[sorted_indices][:top]
    sorted_probs = [f"{100.0 * p:.2f}%" for p in sorted_probs]
    sorted_labels = [class_names[i] for i in sorted_indices][:top]

    return sorted_labels, sorted_probs


# Function to select an image and classify it
def classify_image():
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        print("No image selected.")
        return

    image = preprocess_image(file_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = class_names[predicted.item()]

        print(get_probabilities(output))

    print(f"Predicted Class: {predicted_label}")


if __name__ == "__main__":
    classify_image()
