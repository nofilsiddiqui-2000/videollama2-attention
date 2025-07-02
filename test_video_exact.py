#!/usr/bin/env python3
import os, sys, torch, gc
videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

# Set cache
cache = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
os.environ.update({
    "HF_HOME": cache,
    "TRANSFORMERS_CACHE": cache,
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"  # Prevent fragmentation
})

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from peft import PeftModel

def test_single_prompt(model, processor, tokenizer, video_tensor, prompt):
    """Test a single prompt to avoid memory buildup"""
    try:
        with torch.no_grad():
            # Clear cache before each inference
            torch.cuda.empty_cache()
            
            output = mm_infer(
                video_tensor,     # 1st: image_or_video
                prompt,           # 2nd: instruct  
                model,            # 3rd: model
                tokenizer,        # 4th: tokenizer
                modal='video',    # 5th: modal
                do_sample=False,
                max_new_tokens=20,  # Shorter to save memory
                temperature=0.0
            )
            
            return output.split('\n')[0]  # First line only
            
    except Exception as e:
        print(f"   Error: {e}")
        return "ERROR"
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def test_model(model_name, checkpoint_dir=None):
    """Test model with extreme memory management"""
    print(f"\n🔍 TESTING {model_name.upper()}")
    print("="*50)
    
    # Restart CUDA context to clear fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
    
    gc.collect()
    
    model, base_model, processor, tokenizer, video_tensor = None, None, None, None, None
    
    try:
        disable_torch_init()
        base_model, processor, tokenizer = model_init(
            "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=cache
        )
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        if checkpoint_dir:
            model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            print(f"✅ Enhanced model loaded")
            
            # Convert LoRA to FP32
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()
                    p.requires_grad_(False)
        else:
            model = base_model
            print(f"✅ Baseline model loaded")
        
        model.to("cuda")
        model.eval()
        
        # Pre-process video ONCE
        video_path = "kinetics400_dataset/riding_bike_RgKAFK5djSk_001.mp4"
        video_tensor = processor['video'](video_path).to(torch.float16)
        video_tensor = (video_tensor.clamp(0, 1) * 2 - 1).cuda(non_blocking=True)
        
        print(f"📹 Video tensor loaded: {video_tensor.shape}")
        print(f"💾 CUDA memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Test prompts ONE AT A TIME
        test_prompts = ["danger", "warning", "danger danger danger warning", "This video shows"]
        results = {}
        
        for prompt in test_prompts:
            print(f"\n🔍 Testing: '{prompt}'")
            result = test_single_prompt(model, processor, tokenizer, video_tensor, prompt)
            results[prompt] = result
            if result != "ERROR":
                print(f"✅ '{prompt}' → '{result}'")
            else:
                print(f"❌ '{prompt}' → FAILED")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}
    finally:
        # AGGRESSIVE CLEANUP
        del model, base_model, processor, tokenizer, video_tensor
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def main():
    print("🎯 TESTING WITH AGGRESSIVE MEMORY MANAGEMENT")
    print("="*60)
    
    # Test baseline
    baseline_results = test_model("baseline")
    
    # Force cleanup between models
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    # Test enhanced
    enhanced_results = test_model("enhanced", "massive_epoch_danger")
    
    # Compare results
    print(f"\n🔍 FINAL COMPARISON:")
    print("="*40)
    
    if baseline_results and enhanced_results:
        differences = 0
        for prompt in baseline_results:
            if prompt in enhanced_results:
                baseline_out = baseline_results[prompt]
                enhanced_out = enhanced_results[prompt]
                
                if baseline_out != enhanced_out and baseline_out != "ERROR" and enhanced_out != "ERROR":
                    print(f"\n🎉 DIFFERENT: '{prompt}'")
                    print(f"   Baseline:  {baseline_out}")
                    print(f"   Enhanced:  {enhanced_out}")
                    differences += 1
                else:
                    if baseline_out != "ERROR":
                        print(f"❌ SAME: '{prompt}' → {baseline_out}")
                    else:
                        print(f"❌ ERROR: '{prompt}'")
        
        if differences > 0:
            print(f"\n🔥🔥🔥 SUCCESS! {differences} DIFFERENT OUTPUTS! 🔥🔥🔥")
            print("✅ YOUR 855-STEP TRAINING WORKED!")
        else:
            print(f"\n💔 No differences detected")
    else:
        print("❌ Could not compare - both failed")

if __name__ == "__main__":
    main()
