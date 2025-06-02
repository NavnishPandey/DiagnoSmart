async function getPrediction() {
    const complaint = document.getElementById('complaint').value.trim();
  
    if (!complaint) {
      alert("Please enter your symptoms.");
      return;
    }
  
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ complaint: complaint })
      });
  
      if (!response.ok) throw new Error("Server error");
  
      const data = await response.json();
  
      document.getElementById('specialty').innerText = data.specialty || 'N/A';
      document.getElementById('chronic').innerText = data.chronic_status || 'N/A';
      document.getElementById('severity').innerText = data.severity_level || 'N/A';
      document.getElementById('results').style.display = 'block';
  
    } catch (error) {
      console.error('Prediction failed:', error);
      alert("Failed to get prediction. Check backend connection.");
    }
  }
  