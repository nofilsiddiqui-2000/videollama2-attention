#!/usr/bin/env python3
import os, sys, torch
from pathlib import Path

# Set environment
cache = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
os.environ.update({
    "HF_HOME": cache,
    "TRANSFORMERS_CACHE": cache,
    "TOKENIZERS_PARALLELISM": "false"
})

videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

from videollama2 import model_init
from videollama2.utils import disable_torch_init
from peft import PeftModel

def test_model_responses():
    print("🔍 TESTING ACTUAL MODEL RESPONSES")
    print("="*50)
    
    # Load baseline model
    print("Loading baseline model...")
    disable_torch_init()
    baseline_model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir=os.environ["HF_HOME"]
    )
    baseline_model.to("cuda")
    baseline_model.eval()
    
    # Test baseline
    test_prompts = [
        "danger",
        "warning", 
        "This video shows",
        "Safety:",
    ]
    
    print("\n📊 BASELINE MODEL RESPONSES:")
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = baseline_model.generate(
                    inputs.input_ids, 
                    max_length=inputs.input_ids.shape[1] + 10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated[len(prompt):].strip()
            print(f"'{prompt}' → '{continuation}'")
        except Exception as e:
            print(f"'{prompt}' → ERROR: {str(e)[:50]}")
    
    # Clean up baseline
    del baseline_model
    torch.cuda.empty_cache()
    
    # Load enhanced model
    print("\n⏳ Loading enhanced model...")
    try:
        base_model, _, _ = model_init(
            "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=os.environ["HF_HOME"]
        )
        
        enhanced_model = PeftModel.from_pretrained(
            base_model,
            "explicit_danger_vocab",
            is_trainable=False
        )
        enhanced_model.to("cuda")
        enhanced_model.eval()
        
        print("\n📊 ENHANCED MODEL RESPONSES:")
        for prompt in test_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = enhanced_model.generate(
                        inputs.input_ids, 
                        max_length=inputs.input_ids.shape[1] + 10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                continuation = generated[len(prompt):].strip()
                print(f"'{prompt}' → '{continuation}'")
            except Exception as e:
                print(f"'{prompt}' → ERROR: {str(e)[:50]}")
        
    except Exception as e:
        print(f"❌ Could not load enhanced model: {e}")
    
    print("\n🏁 ACTUAL RESPONSE TEST COMPLETE!")

if __name__ == "__main__":
    test_model_responses()
