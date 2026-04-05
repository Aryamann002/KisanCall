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
    class MockEngine:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, *args, **kwargs):
            return {"text": "dummy text", "language": "Romanized", "latency_ms": 100}
        def classify(self, *args, **kwargs):
            return {"decision": "ALLOW", "is_farming": True, "is_safe": True, "reason": "Mock pass", "latency_ms": 50}
        def route(self, *args, **kwargs):
            return {"intent": "general_farming", "latency_ms": 30}
        def generate(self, *args, **kwargs):
            return {"response": "This is a mock response from the fallback API.", "latency_ms": 250}
            
    STTEngine = GuardrailsEngine = RouterEngine = ResponseEngine = MockEngine

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

class TextQuery(BaseModel):
    query: str

def get_engines():
    if not engines:
        print("Initializing SLM pipeline engines...")
        # Check if we need the Phi-3 base model for ANY engine
        need_phi3 = False
        try:
            # We don't have access to the instances yet, but we know if use_mock=False was hardcoded for any of them
            # GuardrailsEngine, RouterEngine, ResponseEngine are imported.
            # We'll just instantiate them, but handle the base model initialization first.
            engines["stt"] = STTEngine(use_faster_whisper=True) if STTEngine else None
            
            # Since the user might set use_mock=False manually in the code, 
            # we will pre-load the base model if ANY of them are missing it.
            # But wait, it's safer to just instantiate them sequentially. If they don't find the LoRA, they fall back.
            # Let's load the base model ONCE and pass it to all of them.
            base_model = None
            tokenizer = None
            # We unconditionally load the Shared Alpha Phi-3 Base Model
            # to feed into the 3 engines. If LoRA is missing, they will Zero-Shot fallback.
            print("Loading Shared Phi-3 Base Model into VRAM for Zero-Shot & LoRA...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
            
            engines["guardrails"] = GuardrailsEngine(use_mock=False, base_model=base_model, tokenizer=tokenizer) if GuardrailsEngine else None
            engines["router"] = RouterEngine(use_mock=False, base_model=base_model, tokenizer=tokenizer) if RouterEngine else None
            engines["response"] = ResponseEngine(use_mock=False, base_model=base_model, tokenizer=tokenizer) if ResponseEngine else None
            
        except Exception as e:
            print(f"Engine Initialization Error: {e}")
            
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
        "is_farming": guard_res.get("is_farming", True),
        "is_safe": guard_res.get("is_safe", True),
        "reason": guard_res.get("reason", "Checks passed."),
        "decision": "ALLOW",
        "intent": intent,
        "response": final_response,
        "audio_base64": audio_base64,
        "latencies": latencies
    }

@app.post("/process_text")
async def process_text(payload: TextQuery):
    """
    Text-only pipeline bypassing STT:
    Guardrails -> Router -> Response -> TTS
    """
    start_total = time.perf_counter()
    engs = get_engines()
    
    query_text = payload.query
    if not query_text:
        return {"error": "Empty query"}
        
    latencies = {}
    lang = "Romanized" # Default for text
    
    # Check language script roughly
    if any("\u0a00" <= c <= "\u0a7f" for c in query_text):
        lang = "Punjabi"
    elif any("\u0900" <= c <= "\u097f" for c in query_text):
        lang = "Hindi"

    # 1. Guardrails
    guard_res = engs["guardrails"].classify(query_text)
    latencies["guardrails"] = guard_res["latency_ms"]
    
    if guard_res["decision"] == "BLOCK":
        return {
            "text": query_text,
            "decision": "BLOCK",
            "is_farming": guard_res.get("is_farming", False),
            "is_safe": guard_res.get("is_safe", False),
            "reason": guard_res.get("reason", "Unsafe content"),
            "latencies": latencies,
            "response": "ਮਾਫ ਕਰਨਾ, ਮੈਂ ਤੁਹਾਡੀ ਮਦਦ ਨਹੀਂ ਕਰ ਸਕਦਾ। (Sorry, I cannot help with this.)"
        }
    elif guard_res["decision"] == "REDIRECT":
        return {
            "text": query_text,
            "decision": "REDIRECT",
            "is_farming": guard_res.get("is_farming", False),
            "is_safe": guard_res.get("is_safe", True),
            "reason": guard_res.get("reason", "Not farming related"),
            "latencies": latencies,
            "response": "ਮੈਂ ਸਿਰਫ ਖੇਤੀਬਾੜੀ ਬਾਰੇ ਜਾਣਕਾਰੀ ਦੇ ਸਕਦਾ ਹਾਂ। (I can only provide agricultural info.)"
        }
        
    # 2. Router
    router_res = engs["router"].route(query_text)
    latencies["router"] = router_res["latency_ms"]
    intent = router_res["intent"]
    
    # 3. Response
    context = ""
    resp_res = engs["response"].generate(query_text, intent, context)
    latencies["response"] = resp_res["latency_ms"]
    final_response = resp_res["response"]
    
    # 4. TTS fallback
    audio_base64 = None
    try:
        from gtts import gTTS
        tts_start = time.perf_counter()
        lang_code = "pa" if "pa" in lang.lower() or "punjabi" in lang.lower() else "hi"
        tts = gTTS(text=final_response, lang=lang_code)
        import io
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        import base64
        audio_base64 = base64.b64encode(fp.getvalue()).decode()
        latencies["tts"] = round((time.perf_counter() - tts_start) * 1000, 1)
    except Exception as e:
        print(f"TTS Error: {e}")

    total_latency = round((time.perf_counter() - start_total) * 1000, 1)
    latencies["total"] = total_latency

    return {
        "text": query_text,
        "language": lang,
        "is_farming": guard_res.get("is_farming", True),
        "is_safe": guard_res.get("is_safe", True),
        "reason": guard_res.get("reason", "Checks passed."),
        "decision": "ALLOW",
        "intent": intent,
        "response": final_response,
        "audio_base64": audio_base64,
        "latencies": latencies
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
