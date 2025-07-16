import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [complaint, setComplaint] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [touched, setTouched] = useState(false);

  const isValid = complaint.trim().length >= 10;

  const getPrediction = async () => {
    setTouched(true);
    if (!isValid) return;

    setLoading(true);
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ complaint })
      });

      if (!response.ok) throw new Error('Server error');

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert("Failed to get prediction. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center mb-4">Medical Complaint Predictor</h1>
      <p className="text-muted">Describe your symptoms below and get a prediction:</p>

      <div className="mb-3">
        <textarea
          className={`form-control ${touched && !isValid ? 'is-invalid' : ''}`}
          rows="5"
          placeholder="Enter your symptoms here (at least 10 characters)..."
          value={complaint}
          onChange={(e) => setComplaint(e.target.value)}
          onBlur={() => setTouched(true)}
        />
        {touched && !isValid && (
          <div className="invalid-feedback">
            Please enter at least 10 characters describing your symptoms.
          </div>
        )}
      </div>

      <button
        className="btn btn-primary w-100"
        onClick={getPrediction}
        disabled={loading || !isValid}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {result && (
        <div className="card mt-4">
          <div className="card-body">
            <h5 className="card-title">Prediction Results</h5>
            <p><strong>Specialty:</strong> {result.specialty || 'N/A'}</p>
            <p><strong>Chronic Condition:</strong> {result.chronic_status || 'N/A'}</p>
            <p><strong>Severity Level:</strong> {result.severity_level || 'N/A'}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
