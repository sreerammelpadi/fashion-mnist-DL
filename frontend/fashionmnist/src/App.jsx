import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [labels, setLabels] = useState([]);
  const [probabilities, setProbabilities] = useState([]);
  const [error, setError] = useState(null);

  // Handle file input change
  const onFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  // Handle image upload and classification request
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setError(null);  // Reset error
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:7676/classify", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const { prediction, labels, probabilities } = response.data;
      setPrediction(prediction);
      setLabels(labels);
      setProbabilities(probabilities);
    } catch (error) {
      setError("Error uploading image or classifying.");
      console.error(error);
    }
  };

  return (
    <div className="App">
      <h1>Fashion MNIST Classification</h1>
      <h3 style={{ marginBottom: '300px' }}>CSE 676B - DL Code Demo by Sreeram Melpadi</h3>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={onFileChange} />
        <button type="submit">Classify Image</button>
      </form>

      {error && <div style={{ color: 'red' }}>{error}</div>}

      {file && (
        <div>
          <h3>Selected Image:</h3>
          <img
            src={URL.createObjectURL(file)}
            alt="Selected"
            style={{ maxWidth: '300px', maxHeight: '300px' }}
          />
        </div>
      )}

      {prediction && (
        <div>
          <h3>Prediction: {prediction}</h3>
          <h4>Top 3 Labels with probabilities:</h4>
          <ul>
            {labels.map((label, index) => (
              <li key={index}>
                {label}: {probabilities[index]}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;



