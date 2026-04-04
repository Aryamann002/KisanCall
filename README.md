# 🌾 KisanCall — Voice AI for Farmers

**KisanCall** is an offline-first, vernacular Voice AI platform explicitly built for Indian farmers. It operates entirely on consumer-grade hardware (like a 6GB VRAM RTX 3050) completely independent of expensive cloud infrastructure like OpenAI or Anthropic APIs.

Built for the **Eclipse 6.0 Hackathon** by **Team EL BICHO** (Thapar Institute of Engineering & Technology).

---

## 🏗️ Architecture: The 4-SLM Cascade

Rather than relying on one massive billion-parameter model, KisanCall relies on a highly-optimized cascade of 4 Small Language Models (SLMs). This compartmentalized architecture allows parallel processing, extremely fast inference (<500ms latency), and extreme hardware efficiency.

1. **Speech-to-Text (STT) Engine**
   - **Model:** Whisper (Fine-tuned on Punjabi/Hindi via LoRA)
   - **Function:** Converts the farmer's raw audio into text. It uses Silero VAD to detect when speech starts and stops.
2. **Guardrails SLM**
   - **Model:** Phi-3 Mini (4-bit QLoRA)
   - **Function:** Immediately filters queries. If the query isn't farming-related (`REDIRECT`) or contains unsafe/abusive content (`BLOCK`), the pipeline halts and returns a fallback response. This saves the remaining models from wasting compute.
3. **Intent Router SLM**
   - **Model:** Phi-3 Mini (4-bit QLoRA)
   - **Function:** Routes safe queries into specific agricultural domains: `crop_disease`, `mandi_price`, `weather`, or `govt_scheme`.
4. **Response Generator SLM**
   - **Model:** Phi-3 Mini (4-bit QLoRA)
   - **Function:** Acts as the agricultural expert. Ingests context (retrieved from databases/API) alongside the intent to deliver a localized, voice-friendly response.

---

## ⚡ Hardware & Software Stack

- **Hardware Target:** RTX 3050 Laptop (6GB VRAM)
- **Frameworks:** PyTorch (CUDA 12.1), Transformers, PEFT, bitsandbytes
- **Fine-Tuning:** [Oumi](https://oumi.ai) (Hackathon Requirement framework used for pipeline QLoRA training)
- **Backend Server:** FastAPI & Uvicorn
- **Frontend UI:** Vanilla HTML/CSS/JS with intelligent pipeline step-by-step animation. 

---

## 🚀 Running the Project Locally

### 1. Requirements Checklist
You need the following installed:
- Python 3.12+
- `uv` (Fast python package installer)
- Git LFS (Large File System) if downloading raw model weights.

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/your-username/kisancall.git
cd kisancall

# Create a virtual environment using uv
uv venv Backend\venv

# Activate the environment (Windows)
.\Backend\venv\Scripts\Activate.ps1

# Install requirements (Pulls PyTorch 2.5.1 + CUDA 12.1 automatically)
uv pip install -r Backend\requirements.txt
```

### 3. Start the Backend API Pipeline

```powershell
# Navigate to the API and start the FastAPI uvicorn daemon
python -m uvicorn Backend.api.main:app --app-dir . --host 0.0.0.0 --port 8000 --reload
```
*Note: If local trained models are unavailable, the backend will auto-fallback to simulated mock logic so parsing and UI behavior remain uninterrupted.*

### 4. Launch the Frontend
You don't need Node.js or React. Just double-click the `Frontend/index_2.html` file in your browser to interact with the mock phone interface!

---

## 🔮 Roadmap Vision
While currently scoped for the web browser demo, KisanCall is engineered to eventually hook directly into telecom networks via an IVR (Interactive Voice Response) system. This means farmers can simply dial a toll-free number from basic 2G feature phones — zero internet, zero apps, and zero typing required.
