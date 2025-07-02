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

# Enable TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def test_single_model(model_name, checkpoint_dir=None):
    """Test a single model to avoid OOM"""
    print(f"🔍 TESTING {model_name.upper()}")
    print("="*50)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    try:
        disable_torch_init()
        base_model, processor, tokenizer = model_init(
            "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=cache
        )
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print(f"✅ Pad token ID set to: {tokenizer.pad_token_id}")
        
        if checkpoint_dir:
            model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            print(f"✅ Loaded enhanced model from {checkpoint_dir}")
            print(f"   PEFT config: {model.peft_config}")
            
            # Keep LoRA weights in FP32 ONLY
            lora_weights_found = False
            for n, p in model.named_parameters():
                if "lora_" in n:
                    lora_weights_found = True
                    p.data = p.data.float()
                    p.requires_grad_(False)
                    print(f"   LoRA weight: {n} → max={p.abs().max():.6f}")
            
            if not lora_weights_found:
                print("❌ WARNING: No LoRA weights found!")
        else:
            model = base_model
            print("✅ Loaded baseline model")
        
        model.to("cuda")
        model.eval()
        base_model.eval()
        
        # Pre-process video
        video_path = "kinetics400_dataset/riding_bike_RgKAFK5djSk_001.mp4"
        print(f"📹 Pre-processing video: {video_path}")
        
        video_tensor = processor['video'](video_path).to(torch.float16)
        video_tensor = (video_tensor.clamp(0, 1) * 2 - 1).cuda(non_blocking=True)
        print(f"   Video tensor shape: {video_tensor.shape}")
        
        # Test prompts from training
        test_prompts = [
            "danger",
            "warning", 
            "danger danger danger warning",
            "This video shows"
        ]
        
        results = {}
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for prompt in test_prompts:
                print(f"\n🔍 Testing: '{prompt}'")
                
                try:
                    # ✅ CORRECT SIGNATURE: mm_infer(image_or_video, instruct, model, tokenizer, modal='video')
                    output = mm_infer(
                        video_tensor,     # 1st: image_or_video
                        prompt,           # 2nd: instruct  
                        model,            # 3rd: model
                        tokenizer,        # 4th: tokenizer
                        modal='video',    # 5th: modal (keyword)
                        do_sample=False,
                        max_new_tokens=40,
                        temperature=0.0
                    )
                    
                    results[prompt] = output
                    continuation = output.split('\n')[0]
                    print(f"✅ '{prompt}' → '{continuation}'")
                    
                except Exception as infer_error:
                    print(f"❌ mm_infer error: {infer_error}")
                    results[prompt] = "ERROR"
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'video_tensor' in locals():
            del video_tensor
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def main():
    print("🎯 TESTING VIDEO+TEXT CONTEXT (CORRECT mm_infer SIGNATURE!)")
    print("="*70)
    
    # Test baseline first
    baseline_results = test_single_model("baseline")
    
    print("\n" + "="*70)
    
    # Test enhanced model
    enhanced_results = test_single_model("enhanced", "massive_epoch_danger")
    
    # Compare results
    print("\n🔍 FINAL COMPARISON:")
    print("="*50)
    
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
                    print("-" * 50)
                    differences += 1
                else:
                    if baseline_out != "ERROR":
                        print(f"❌ SAME: '{prompt}' → {baseline_out}")
                    else:
                        print(f"❌ ERROR: '{prompt}' → both failed")
        
        if differences > 0:
            print(f"\n🔥🔥🔥 SUCCESS! {differences} DIFFERENT OUTPUTS! 🔥🔥🔥")
            print("✅ THE 855-STEP MASSIVE EPOCH TRAINING WORKED!")
            print("✅ LoRA adapter is loaded and changing behavior!")
        else:
            print(f"\n💔 RESULT: All outputs identical")
            if any(v != "ERROR" for v in baseline_results.values()):
                print("❌ The massive training didn't change video+text behavior")
                print("❌ May need different LoRA configuration")
            else:
                print("❌ Both models failed - check setup")
    else:
        print("❌ Could not compare - loading failed")

if __name__ == "__main__":
    main()
