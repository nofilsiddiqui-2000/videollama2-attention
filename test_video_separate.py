#!/usr/bin/env python3
import os, sys, torch, gc
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

def test_single_model(model_name, checkpoint_dir=None):
    """Test a single model to avoid OOM"""
    print(f"🔍 TESTING {model_name.upper()}")
    print("="*50)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        disable_torch_init()
        base_model, processor, tokenizer = model_init(
            "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=cache
        )
        
        if checkpoint_dir:
            model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=False)
            print(f"✅ Loaded enhanced model from {checkpoint_dir}")
        else:
            model = base_model
            print("✅ Loaded baseline model")
        
        model.to("cuda")
        model.eval()
        
        # Test video
        video_path = "kinetics400_dataset/riding_bike/riding_bike_RgKAFK5djSk_001.mp4"
        video_tensor = processor["video"](video_path)
        video_tensor = video_tensor.clamp(0, 1) * 2 - 1
        video_tensor = video_tensor.to("cuda", dtype=torch.float16)
        
        # Test prompts from training
        test_prompts = [
            "danger",
            "warning", 
            "danger danger danger warning",
            "This video shows"
        ]
        
        results = {}
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=32).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=video_tensor.unsqueeze(0),
                    max_length=inputs.input_ids.shape[1] + 8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated_text[len(prompt):].strip()
            results[prompt] = continuation
            print(f"'{prompt}' → '{continuation}'")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'video_tensor' in locals():
            del video_tensor
        torch.cuda.empty_cache()
        gc.collect()

def main():
    print("🎯 TESTING VIDEO+TEXT CONTEXT (Memory Safe)")
    print("="*60)
    
    # Test baseline first
    baseline_results = test_single_model("baseline")
    
    print("\n" + "="*60)
    
    # Test enhanced model
    enhanced_results = test_single_model("enhanced", "massive_epoch_danger")
    
    # Compare results
    print("\n🔍 COMPARISON:")
    print("="*40)
    
    if baseline_results and enhanced_results:
        differences = 0
        for prompt in baseline_results:
            if prompt in enhanced_results:
                baseline_out = baseline_results[prompt]
                enhanced_out = enhanced_results[prompt]
                
                if baseline_out != enhanced_out:
                    print(f"✅ DIFFERENT: '{prompt}'")
                    print(f"   Baseline:  '{baseline_out}'")
                    print(f"   Enhanced:  '{enhanced_out}'")
                    differences += 1
                else:
                    print(f"❌ SAME: '{prompt}' → '{baseline_out}'")
        
        if differences > 0:
            print(f"\n🎉 SUCCESS: {differences} different outputs detected!")
        else:
            print(f"\n💔 FAILURE: All outputs identical")
    else:
        print("❌ Could not compare - loading failed")

if __name__ == "__main__":
    main()
