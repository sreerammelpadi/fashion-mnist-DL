"""
This file is for testing the local API to see if the API works as expected.
This file acts as a dummy front end API caller.
"""

import requests
import tkinter as tk
from tkinter import filedialog

# URL of the API endpoint
url = 'http://127.0.0.1:7676/classify'

# image prompter
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Select an Image",
                                        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

with open(image_path, 'rb') as img_file:
    files = {'file': img_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Response from server:")
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
