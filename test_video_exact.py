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
        
        # FIX 3: Set padding token
        tokenizer.pad_token = tokenizer.eos_token
        
        if checkpoint_dir:
            model = PeftModel.from_pretrained(
                base_model, 
                checkpoint_dir, 
                adapter_name="default",  # FIX 5: Explicit adapter name
                is_trainable=False
            )
            print(f"✅ Loaded enhanced model from {checkpoint_dir}")
            
            # FIX 6: Verify LoRA weights are loaded
            lora_weights_found = False
            for n, p in model.named_parameters():
                if 'lora_' in n:
                    lora_weights_found = True
                    print(f"   LoRA weight: {n} → max={p.abs().max():.6f}")
                    # FIX 8: LoRA matrices to FP32
                    p.data = p.data.float()
            
            if not lora_weights_found:
                print("❌ WARNING: No LoRA weights found!")
        else:
            model = base_model
            print("✅ Loaded baseline model")
        
        model.to("cuda")
        # FIX 7: Eval mode on both
        model.eval()
        base_model.eval()
        
        # FIX 2: Pre-process video to tensor ONCE
        video_path = "kinetics400_dataset/riding_bike_RgKAFK5djSk_001.mp4"
        print(f"📹 Pre-processing video: {video_path}")
        
        video_tensor = processor['video'](video_path)  # Decode on CPU
        video_tensor = video_tensor.clamp(0, 1).to('cuda', torch.float16)
        print(f"   Video tensor shape: {video_tensor.shape}")
        
        # Test prompts from training
        test_prompts = [
            "danger",
            "warning", 
            "danger danger danger warning",
            "This video shows"
        ]
        
        results = {}
        for prompt in test_prompts:
            print(f"\n🔍 Testing: '{prompt}'")
            
            try:
                # FIX 1: Correct mm_infer signature with keyword args
                output = mm_infer(
                    model,                    # required
                    processor,                # required  
                    video_tensor,             # **tensor**, not path
                    prompt,                   # instruction
                    modal='video',            # keyword arg
                    tokenizer=tokenizer,      # keyword arg
                    do_sample=False           # keyword arg
                )
                
                results[prompt] = output
                print(f"✅ '{prompt}' → '{output[:150]}...'")
                
            except Exception as infer_error:
                print(f"❌ mm_infer error: {infer_error}")
                
                # FIX 9: Fallback to direct generate() for sanity check
                try:
                    print("   Trying direct generate()...")
                    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
                    
                    with torch.no_grad():
                        out_ids = model.generate(
                            pixel_values=video_tensor.unsqueeze(0),
                            input_ids=inputs.input_ids,
                            max_new_tokens=32,
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    output = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    continuation = output[len(prompt):].strip()
                    results[prompt] = continuation
                    print(f"✅ '{prompt}' (direct) → '{continuation}'")
                    
                except Exception as gen_error:
                    print(f"❌ Direct generate error: {gen_error}")
                    results[prompt] = "ERROR"
            
            # FIX 4: Free memory between iterations
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        # FIX 4: Aggressive cleanup
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'video_tensor' in locals():
            del video_tensor
        torch.cuda.empty_cache()
        gc.collect()

def main():
    print("🎯 TESTING VIDEO+TEXT CONTEXT (ALL FIXES APPLIED)")
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
            print("✅ Video+text context shows clear behavioral changes!")
        else:
            print(f"\n💔 RESULT: All outputs identical")
            if any(v != "ERROR" for v in baseline_results.values()):
                print("❌ The massive training didn't change video+text behavior")
                print("❌ May need different LoRA configuration or more training")
            else:
                print("❌ Both models failed - technical issues need fixing")
    else:
        print("❌ Could not compare - loading failed")

if __name__ == "__main__":
    main()
