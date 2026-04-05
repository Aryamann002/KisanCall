import time
from typing import Dict, Any

class ResponseEngine:
    """
    Response Generator Inference Engine using Phi-3 mini.
    Generates concise agricultural advice in the user's language based on intent.
    """
    
    def __init__(self, model_dir: str = "response/model/response-phi3-mini", use_mock: bool = False, base_model=None, tokenizer=None):
        self.use_mock = use_mock
        if self.use_mock:
            print("ResponseEngine initialized in MOCK mode")
            return
            
        print(f"Loading Response Generator SLM from {model_dir}...")
        try:
            # Load LoRA adapters or fallback to ZERO-SHOT base model
            import os
            if os.path.isdir(model_dir):
                self.model = PeftModel.from_pretrained(base_model, model_dir)
                print("Response SLM loaded with LoRA successfully.")
            else:
                self.model = base_model
                print(f"LoRA directory missing ({model_dir}). Using ZERO-SHOT Base Phi-3 for ResponseEngine!")

            self.model.eval()
        except Exception as e:
            print(f"Failed to load Response SLM: {e}")
            self.use_mock = True

    def generate(self, query: str, intent: str, context: str = "") -> Dict[str, Any]:
        """
        Generates the voice-friendly response.
        """
        start = time.perf_counter()
        
        if self.use_mock:
            if intent == "crop_disease":
                resp = "ਇਸ ਬਿਮਾਰੀ ਲਈ ਪ੍ਰੋਪੀਕੋਨਾਜ਼ੋਲ ੨੫% ਈ.ਸੀ. ਦੀ ਸਪਰੇਅ ਕਰੋ।"
            elif intent == "mandi_price":
                resp = "ਅੱਜ ਮੰਡੀ ਵਿੱਚ ਝੋਨੇ ਦਾ ਭਾਅ 2200 ਰੁਪਏ ਪ੍ਰਤੀ ਕੁਇੰਟਲ ਹੈ।"
            elif intent == "weather":
                resp = "ਆਉਣ ਵਾਲੇ ਦੋ ਦਿਨਾਂ ਵਿੱਚ ਮੀਂਹ ਪੈਣ ਦੀ ਸੰਭਾਵਨਾ ਹੈ, ਕਿਰਪਾ ਕਰਕੇ ਸਪਰੇਅ ਨਾ ਕਰੋ।"
            else:
                resp = "ਪੀਐਮ ਕਿਸਾਨ ਯੋਜਨਾ ਤਹਿਤ ਤੁਹਾਨੂੰ ੬੦੦੦ ਰੁਪਏ ਸਾਲਾਨਾ ਮਿਲਣਗੇ।"
        else:
            prompt = f"<|system|>\nYou are an agricultural advisor. Generate a concise, simple, spoken response to the farmer's query. Tone should be helpful and match the language of the query. Do not use markdown. Intent: {intent}. Additional Context: {context}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                )
                
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            resp = response

        latency = round((time.perf_counter() - start) * 1000, 1)
        return {
            "response": resp,
            "latency_ms": latency
        }

if __name__ == "__main__":
    engine = ResponseEngine(use_mock=True)
    res = engine.generate("kਣਕ ਪੀਲੀ ਹੋ ਰਹੀ ਹੈ", "crop_disease")
    print(res)
