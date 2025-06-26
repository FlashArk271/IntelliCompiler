
# 🧠 IntelliCompiler — AI-powered Online Code Compiler

**IntelliCompiler** is a full-stack, AI-integrated online code execution platform developed using **Flask**, **CodeMirror**, and **Groq API**, supporting dynamic, multi-language code execution with automatic language detection and AI-powered **Time and Space Complexity Analysis**.

## 🔥 Features

* 🔍 **Automatic Language Detection** (C, C++, Java, Python)
* ⚙️ **Real-time Code Execution** with User Input Support
* 📊 **AI-Powered Time and Space Complexity Analysis**
* 🌐 **RESTful API** powered backend using Flask
* ⚡ Frontend built using **HTML**, **TailwindCSS**, and **Vanilla JS**
* 💡 Dynamic Input/Output Interaction
* 🧪 Built-in Health & Status Endpoints

---


---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/FlashArk271/IntelliCompiler.git
cd IntelliCompiler
```

### 2. Setup Backend (Flask + Groq)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory and add:

```
GROQ_API_KEY=your_groq_api_key_here
```

Then run:

```bash
python app.py
```

Flask backend will be available at:
📍 **[http://localhost:5000](http://localhost:5000)**

---

### 3. Setup Frontend (served separately)

In another terminal, navigate to the folder where your frontend is located(remain in the same folder if app.py and frontend files in same folder):

```bash
cd frontend/
python -m http.server 8000
```

Frontend will now be served at:
🌐 **[http://localhost:8000](http://localhost:8000)**

---

## 🔌 API Endpoints

| Endpoint                   | Description                             |
| -------------------------- | --------------------------------------- |
| `POST /execute-code`       | Execute code with optional input        |
| `POST /send-input`         | Send input to running process           |
| `POST /get-status`         | Get live execution status               |
| `POST /stop-process`       | Force-stop a running process            |
| `POST /detect-language`    | Auto-detect code language               |
| `POST /analyze-complexity` | Get time & space complexity via Groq AI |
| `GET /health`              | Server health status                    |

---

## 🧠 AI-Powered Complexity Analysis

Using the **Groq API** and an expert system prompt, the backend provides:

* ✅ **Time & Space Complexity** in Big-O notation
* 📖 Detailed **Explanations** for each
* 🧮 Best, Average, Worst Case breakdown
* 🧠 List of **Dominant Operations**

> Achieves **90%+ accuracy** across standard algorithmic patterns.

---

## 🛠️ Tech Stack

* **Frontend:** HTML, TailwindCSS, JavaScript
* **Backend:** Python, Flask
* **Editor:** CodeMirror (JS)
* **AI Model:** Groq (LLaMA 3.1 8B Instant)
* **Code Execution:** Python `subprocess`, Tempfile-based
* **Language Detection:** Regex pattern matching
* **Security:** Process isolation, UUID-based execution IDs

---


