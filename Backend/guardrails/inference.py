import json
import time
from typing import Dict, Any

class GuardrailsEngine:
    """
    Guardrails inference engine using fine-tuned Phi-3 mini.
    Uses Oumi framework for inference if available.
    """
    
    def __init__(self, model_dir: str = "guardrails/model/guardrails-phi3-mini", use_mock: bool = False, base_model=None, tokenizer=None):
        self.use_mock = use_mock
        if self.use_mock:
            print("GuardrailsEngine initialized in MOCK mode (faster testing)")
            return
            
        print(f"Loading Guardrails SLM from {model_dir}...")
        try:
            # Load LoRA adapters or fallback to ZERO-SHOT base model
            import os
            if os.path.isdir(model_dir):
                self.model = PeftModel.from_pretrained(base_model, model_dir)
                print("Guardrails SLM loaded with LoRA successfully.")
            else:
                self.model = base_model
                print(f"LoRA directory missing ({model_dir}). Using ZERO-SHOT Base Phi-3 for Guardrails!")
            
            self.model.eval()
        except Exception as e:
            print(f"Failed to load Guardrails SLM: {e}")
            print("Falling back to MOCK mode.")
            self.use_mock = True

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies a query as ALLOW, REDIRECT, or BLOCK.
        """
        start = time.perf_counter()
        
        if self.use_mock:
            # Mock logic matching the rules without running the heavy model
            lower_q = query.lower()
            if any(word in lower_q for word in ["kill", "bomb", "poison", "illegal", "मार", "बम"]):
                result = {"is_farming": False, "is_safe": False, "decision": "BLOCK"}
            elif any(word in lower_q for word in ["kanak", "jhone", "mandi", "fasal", "kisan", "crop", "wheat", "कणक", "धान", "फसल"]):
                result = {"is_farming": True, "is_safe": True, "decision": "ALLOW"}
            else:
                result = {"is_farming": False, "is_safe": True, "decision": "REDIRECT"}
        else:
            # Actual SLM inference
            prompt = f"<|system|>\nYou are a Guardrails classifier for KisanCall, an agricultural voice assistant. Classify the user query and output a JSON with is_farming (boolean), is_safe (boolean), and decision ('ALLOW', 'REDIRECT', 'BLOCK').<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,  # Low temp for deterministic JSON output
                    do_sample=False,
                )
                
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            try:
                # Find JSON block in the output
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(response[json_start:json_end])
                else:
                    raise ValueError("No JSON found in response")
            except Exception as e:
                print(f"Error parsing Guardrails output: {e}. Raw: {response}")
                # Fallback conservative decision
                result = {"is_farming": False, "is_safe": True, "decision": "REDIRECT"}

        latency = round((time.perf_counter() - start) * 1000, 1)
        result["latency_ms"] = latency
        return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="ਕਣਕ ਦੀ ਫ਼ਸਲ ਪੀਲੀ ਹੋ ਰਹੀ ਹੈ", help="Query to test")
    parser.add_argument("--mock", action="store_true", help="Use mock engine")
    args = parser.parse_args()
    
    engine = GuardrailsEngine(use_mock=args.mock)
    result = engine.classify(args.query)
    print(f"\nQuery: {args.query}")
    print(f"Classification: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()
