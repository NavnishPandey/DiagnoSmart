# 🧠 DiagnoSmart

**DiagnoSmart** is an AI-powered healthcare assistant that analyzes patient complaints to predict:
- The most appropriate **medical specialization** (e.g., Cardiologist, Dermatologist),
- The **severity** of the condition (e.g., mild, moderate, severe), and
- Whether the condition is likely **chronic** or **acute**.

This system helps streamline clinical triage, saving time for healthcare providers and improving patient direction.

---

## 🚀 Features

- 🏥 Predicts medical specialty based on patient complaints
- 📊 Assesses severity and chronicity of symptoms
- 🤖 Uses an attention-based deep learning model
- 🌐 Web interface powered by Flask
- 📦 Trained model and vectorizers saved for deployment

## 🛠️ How the Model Works

### 🔄 Preprocessing

- Patient complaints (free-text) are preprocessed:
  - Lowercased
  - Tokenized
  - Cleaned of punctuation/special characters
- TF-IDF Vectorization is used to convert text into numerical feature vectors using `TfidfVectorizer` from scikit-learn.

### 🧠 Model Architecture

- **Input**: TF-IDF vectorized text
- **Shared Layers**:
  - Fully Connected Layers
  - **Custom Attention Layer** (defined in `attention.py`)
- **Output Heads**:
  - **Medical Specialty** (multi-class classification)
  - **Severity Level** (multi-class classification)
  - **Chronicity** (binary classification)

### ⚙️ Training

- Loss: Combined CrossEntropy losses for each head (with weighting)
- Optimizer: Adam
- Training is done via PyTorch with modular structure in `trainer.py`

---

## 🧪 Evaluation

- Metrics:
  - Accuracy for each output
  - Classification reports
- Model is saved as `Diagnosmart_model.pt`
- Encoders for labels and TF-IDF vectorizer saved as `.pkl` files

---

## 🌐 Web App (Flask UI)

- Flask app (`app.py`) serves the model through a REST API
- User inputs a complaint via a web form
- The backend:
  - Preprocesses input
  - Uses TF-IDF vectorizer
  - Loads and runs the model
  - Returns predicted specialty, severity, and chronicity
- Frontend UI located in the `Frontend/` directory

---

## 📌 Future Enhancements

- Add multilingual complaint processing
- Expand symptom database
- Integrate with EHR systems
- Deploy via Docker or cloud services (e.g., Heroku, AWS)         



