# 🌍 PDF chatbot

## 🛠 Installation & Setup

### 1️⃣ Setting up **environments**

📌 **Prerequisite**: Install [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html).

#### Steps to set up the environment:

```sh
# Create a Conda environment
conda create --name qa_chatbot python=3.11 -y

# Activate the environment
conda activate qa_chatbot

# Install required dependencies
conda install requirements.txt
```
---
---

## 🚀 Running the Application

### 🔥 Start Backend
```sh
# Navigate to the backend directory
cd backend

# Activate the Conda environment
conda activate translate_app

# Run the backend server
python app.py
```

### 🎨 Start Frontend
```sh
# Open VS Code
# In the terminal, start React
npm start
```

---

## 📝 Notes

- Make sure to **activate the Conda environment** before running the backend.
- Ensure the backend is running before launching the frontend.
- If necessary, update the API URL in the frontend (`src/app.js`) to match the correct backend address, e.g., `http://localhost:8000/translate/`.

---
