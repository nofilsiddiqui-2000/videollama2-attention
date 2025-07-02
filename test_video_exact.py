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

from videollama2 import model_init, mm_infer
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
        
        # Use EXACT file path
        video_path = "kinetics400_dataset/riding_bike_RgKAFK5djSk_001.mp4"
        print(f"📹 Testing with: {video_path}")
        
        # Test prompts from training
        test_prompts = [
            "danger",
            "warning", 
            "danger danger danger warning",
            "This video shows"
        ]
        
        results = {}
        for prompt in test_prompts:
            print(f"🔍 Testing: '{prompt}'")
            
            try:
                # ✅ CORRECT SIGNATURE: mm_infer(model, processor, video, tokenizer, instruct, modal)
                output = mm_infer(
                    model,
                    processor,
                    video_path,    # 3rd: video path
                    tokenizer,     # 4th: tokenizer (WAS MISSING!)
                    prompt,        # 5th: instruction
                    modal='video'  # 6th: modal type
                )
                
                results[prompt] = output
                print(f"✅ '{prompt}' → '{output[:100]}...'")
                
            except Exception as infer_error:
                print(f"❌ mm_infer error for '{prompt}': {infer_error}")
                results[prompt] = "ERROR"
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        torch.cuda.empty_cache()
        gc.collect()

def main():
    print("🎯 TESTING VIDEO+TEXT CONTEXT (CORRECT mm_infer SIGNATURE)")
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
                
                if baseline_out != enhanced_out and baseline_out != "ERROR" and enhanced_out != "ERROR":
                    print(f"✅ DIFFERENT: '{prompt}'")
                    print(f"   Baseline:  '{baseline_out[:100]}...'")
                    print(f"   Enhanced:  '{enhanced_out[:100]}...'")
                    differences += 1
                else:
                    if baseline_out != "ERROR":
                        print(f"❌ SAME: '{prompt}' → '{baseline_out[:100]}...'")
                    else:
                        print(f"❌ ERROR: '{prompt}' → both failed")
        
        if differences > 0:
            print(f"\n🎉 SUCCESS: {differences} different outputs detected!")
            print("🔥 THE 855-STEP MASSIVE EPOCH TRAINING WORKED!")
        else:
            print(f"\n💔 FAILURE: All outputs identical")
            print("❌ The massive training didn't change video+text behavior")
    else:
        print("❌ Could not compare - loading failed")

if __name__ == "__main__":
    main()
