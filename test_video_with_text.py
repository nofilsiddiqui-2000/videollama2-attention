#!/usr/bin/env python3
import os, sys, torch
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

# Set cache
cache = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
os.environ.update({
    "HF_HOME": cache,
    "TRANSFORMERS_CACHE": cache,
    "TOKENIZERS_PARALLELISM": "false"
})

from videollama2 import model_init
from videollama2.utils import disable_torch_init
from peft import PeftModel

def test_video_context():
    print("🔍 TESTING VIDEO + TEXT CONTEXT (Actual Training Setup)")
    print("="*60)
    
    # Load baseline
    print("Loading baseline model...")
    disable_torch_init()
    baseline_model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir=cache
    )
    baseline_model.to("cuda")
    baseline_model.eval()
    
    # Load enhanced
    print("Loading enhanced model...")
    enhanced_base, _, _ = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir=cache
    )
    enhanced_model = PeftModel.from_pretrained(
        enhanced_base, 
        "massive_epoch_danger",
        is_trainable=False
    )
    enhanced_model.to("cuda")
    enhanced_model.eval()
    
    # Test video
    video_path = "kinetics400_dataset/riding_bike/riding_bike_RgKAFK5djSk_001.mp4"
    print(f"📹 Testing with: {video_path}")
    
    try:
        video_tensor = processor["video"](video_path)
        video_tensor = video_tensor.clamp(0, 1) * 2 - 1
        video_tensor = video_tensor.to("cuda", dtype=torch.float16)
        
        # Test with training-style prompts
        test_prompts = [
            "danger danger danger warning",
            "warning alert risk unsafe",
            "This video shows dangerous"
        ]
        
        for prompt in test_prompts:
            print(f"\n🎯 Testing prompt: '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=32).to("cuda")
            
            # Test baseline
            with torch.no_grad():
                baseline_out = baseline_model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=video_tensor.unsqueeze(0),
                    max_length=inputs.input_ids.shape[1] + 8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
            
            # Test enhanced  
            with torch.no_grad():
                enhanced_out = enhanced_model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=video_tensor.unsqueeze(0),
                    max_length=inputs.input_ids.shape[1] + 8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            enhanced_text = tokenizer.decode(enhanced_out[0], skip_special_tokens=True)
            
            print(f"📊 BASELINE:  {baseline_text}")
            print(f"📊 ENHANCED:  {enhanced_text}")
            print(f"🔄 DIFFERENT: {baseline_text != enhanced_text}")
            
            if baseline_text != enhanced_text:
                print(f"✅ SUCCESS: Models generate different outputs!")
            else:
                print(f"❌ FAILED: Still identical outputs")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🏁 VIDEO+TEXT CONTEXT TEST COMPLETE!")

if __name__ == "__main__":
    test_video_context()
