import requests
import tkinter as tk
from tkinter import filedialog

# URL of the API endpoint
url = 'http://127.0.0.1:7676/classify'

# Path to your image file
root = tk.Tk()
root.withdraw()  # Hide the Tkinter window
image_path = filedialog.askopenfilename(title="Select an Image",
                                        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

# Open the image in binary mode and send the request
with open(image_path, 'rb') as img_file:
    files = {'file': img_file}
    response = requests.post(url, files=files)

# Print the response from the server
if response.status_code == 200:
    print("Response from server:")
    print(response.json())  # This will print the JSON response from the API
else:
    print(f"Error: {response.status_code}, {response.text}")
