import io
import time
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import engines, but allow running even if dependencies are missing (for setup)
try:
    from stt.inference import STTEngine
    from guardrails.inference import GuardrailsEngine
    from router.inference import RouterEngine
    from response.inference import ResponseEngine
except ImportError as e:
    print(f"Warning: Missing dependencies for engines: {e}")
    STTEngine = GuardrailsEngine = RouterEngine = ResponseEngine = None

app = FastAPI(title="KisanCall Backend API", description="Offline 4-SLM Cascade Pipeline")

# Allow Frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines lazily
engines = {}

def get_engines():
    if not engines:
        print("Initializing SLM pipeline engines...")
        # Start in MOCK mode first if we don't have the fully fine-tuned models
        # During the actual execution with models, use_mock=False
        engines["stt"] = STTEngine(use_faster_whisper=True) if STTEngine else None
        engines["guardrails"] = GuardrailsEngine(use_mock=True) if GuardrailsEngine else None
        engines["router"] = RouterEngine(use_mock=True) if RouterEngine else None
        engines["response"] = ResponseEngine(use_mock=True) if ResponseEngine else None
    return engines

@app.on_event("startup")
async def startup_event():
    # Only try to load models at startup if we want to pre-warm the GPU
    # get_engines()
    pass

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/process_audio")
async def process_audio(audio: UploadFile = File(...)):
    """
    Main pipeline:
    Audio -> STT -> Guardrails -> Router -> Response -> TTS
    """
    start_total = time.perf_counter()
    engs = get_engines()
    
    # Check if STTEngine is available
    if not engs["stt"]:
        return {"error": "STTEngine not initialized or missing dependencies"}
        
    # Read Audio 
    audio_bytes = await audio.read()
    latencies = {}
    
    # 1. STT
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        stt_res = engs["stt"].transcribe(audio_path=tmp_path)
        latencies["stt"] = stt_res["latency_ms"]
        query_text = stt_res["text"]
        lang = stt_res["language"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not query_text:
        return {"error": "No speech detected", "latencies": latencies}

    # 2. Guardrails
    guard_res = engs["guardrails"].classify(query_text)
    latencies["guardrails"] = guard_res["latency_ms"]
    
    if guard_res["decision"] == "BLOCK":
        return {
            "text": query_text,
            "decision": "BLOCK",
            "latencies": latencies,
            "response": "ਮਾਫ ਕਰਨਾ, ਮੈਂ ਤੁਹਾਡੀ ਮਦਦ ਨਹੀਂ ਕਰ ਸਕਦਾ। (Sorry, I cannot help with this.)"
        }
    elif guard_res["decision"] == "REDIRECT":
        return {
            "text": query_text,
            "decision": "REDIRECT",
            "latencies": latencies,
            "response": "ਮੈਂ ਸਿਰਫ ਖੇਤੀਬਾੜੀ ਬਾਰੇ ਜਾਣਕਾਰੀ ਦੇ ਸਕਦਾ ਹਾਂ। (I can only provide agricultural info.)"
        }
        
    # 3. Router
    router_res = engs["router"].route(query_text)
    latencies["router"] = router_res["latency_ms"]
    intent = router_res["intent"]
    
    # 4. Response
    # In a real app we'd fetch context from FAISS or APIs based on intent
    context = ""
    resp_res = engs["response"].generate(query_text, intent, context)
    latencies["response"] = resp_res["latency_ms"]
    final_response = resp_res["response"]
    
    # 5. TTS fallback to gTTS just in case
    # Convert text to audio response if needed
    audio_base64 = None
    try:
        from gtts import gTTS
        tts_start = time.perf_counter()
        lang_code = "pa" if "pa" in lang or "punjabi" in lang else "hi"
        tts = gTTS(text=final_response, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        audio_base64 = base64.b64encode(fp.getvalue()).decode()
        latencies["tts"] = round((time.perf_counter() - tts_start) * 1000, 1)
    except Exception as e:
        print(f"TTS Error: {e}")

    total_latency = round((time.perf_counter() - start_total) * 1000, 1)
    latencies["total"] = total_latency

    return {
        "text": query_text,
        "language": lang,
        "decision": "ALLOW",
        "intent": intent,
        "response": final_response,
        "audio_base64": audio_base64,
        "latencies": latencies
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
