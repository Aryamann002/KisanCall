import time
from typing import Dict, Any
import json

class RouterEngine:
    """
    Intent Router Inference Engine using Phi-3 mini.
    Classifies queries into: crop_disease, weather, mandi_price, or govt_scheme.
    """
    
    def __init__(self, model_dir: str = "router/model/router-phi3-mini", use_mock: bool = False):
        self.use_mock = use_mock
        if self.use_mock:
            print("RouterEngine initialized in MOCK mode")
            return
            
        print(f"Loading Intent Router SLM from {model_dir}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
            
            self.model = PeftModel.from_pretrained(base_model, model_dir)
            self.model.eval()
            print("Router SLM loaded successfully.")
        except Exception as e:
            print(f"Failed to load Router SLM: {e}")
            self.use_mock = True

    def route(self, query: str) -> Dict[str, Any]:
        """
        Routes a query to one of the 4 agricultural capabilities.
        """
        start = time.perf_counter()
        
        if self.use_mock:
            lower_q = query.lower()
            if any(w in lower_q for w in ["rate", "bhav", "mandi", "price", "kiemat"]):
                intent = "mandi_price"
            elif any(w in lower_q for w in ["mausam", "weather", "barsaat", "rain", "thund"]):
                intent = "weather"
            elif any(w in lower_q for w in ["yojana", "scheme", "subsidy", "pm kisan"]):
                intent = "govt_scheme"
            else:
                intent = "crop_disease"
        else:
            prompt = f"<|system|>\nYou are an Intent Router for KisanCall. Classify the user query into exactly one of: 'crop_disease', 'weather', 'mandi_price', or 'govt_scheme'. Output only the category string.<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                )
                
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Match to allowed
            allowed = ["crop_disease", "weather", "mandi_price", "govt_scheme"]
            intent = response if response in allowed else "crop_disease"  # Default

        latency = round((time.perf_counter() - start) * 1000, 1)
        return {
            "intent": intent,
            "latency_ms": latency
        }

if __name__ == "__main__":
    engine = RouterEngine(use_mock=True)
    res = engine.route("aaj da mausam kiven hai?")
    print(res)
