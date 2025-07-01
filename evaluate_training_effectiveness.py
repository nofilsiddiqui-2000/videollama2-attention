#!/usr/bin/env python3
# EVALUATE TRAINING EFFECTIVENESS - Comprehensive evaluation for enhanced VBAD training

import os, sys, json, argparse, gc
from pathlib import Path
import torch
import torch.nn.functional as F
from datetime import datetime
import math

videollama_path = "/nfs/speed-scratch/m_s55102/videollama2-attention/VideoLLaMA2"
sys.path.insert(0, videollama_path)

from videollama2 import model_init
from videollama2.utils import disable_torch_init
from peft import LoraConfig, get_peft_model

def set_env():
    cache = "/nfs/speed-scratch/nofilsiddiqui-2000/hf_cache"
    os.environ.update({
        "HF_HOME": cache,
        "TRANSFORMERS_CACHE": cache,
        "TOKENIZERS_PARALLELISM": "false"
    })
    Path(cache).mkdir(parents=True, exist_ok=True)

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def convert_lora_to_fp32(model):
    """Convert LoRA parameters to FP32"""
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param.data = param.data.to(torch.float32)

def get_danger_keywords():
    """Get danger-related keywords to analyze token improvements"""
    return [
        'danger', 'warning', 'alert', 'risk', 'hazard', 'unsafe', 'caution',
        'threat', 'emergency', 'critical', 'safety', 'accident', 'harm',
        'dangerous', 'risky', 'hazardous', 'perilous', 'threatening'
    ]

def analyze_danger_keywords(tokenizer, top_tokens, top_probs):
    """Analyze presence of danger-related keywords in top predictions"""
    danger_keywords = get_danger_keywords()
    danger_score = 0.0
    danger_count = 0
    
    for i, token in enumerate(top_tokens):
        token_clean = token.strip().lower()
        if any(keyword in token_clean for keyword in danger_keywords):
            danger_count += 1
            # Weight by probability
            danger_score += top_probs[i] if i < len(top_probs) else 0.0
    
    return {
        'danger_token_count': danger_count,
        'danger_score': danger_score,
        'danger_keywords_found': [token for token in top_tokens if any(kw in token.strip().lower() for kw in danger_keywords)]
    }

def comprehensive_model_test(model, processor, tokenizer, video_path, model_name, config=None):
    """Comprehensive model testing including loss, tokens, and danger keyword analysis"""
    try:
        print(f"    🎬 Processing: {os.path.basename(video_path)}")
        
        # Process video (same as training)
        video_tensor = processor["video"](video_path)
        if video_tensor is None:
            return {"error": "Video processing failed", "model_healthy": False}
        
        # Memory check
        memory_bytes = video_tensor.numel() * 2
        if memory_bytes > 200_000_000:
            return {"error": "Video too large", "model_healthy": False}
        
        video_tensor = video_tensor.clamp(0, 1) * 2 - 1
        video_tensor = video_tensor.to("cuda", dtype=torch.float16)
        
        # Test with multiple captions for comprehensive analysis
        test_captions = [
            "This video shows various activities - danger warning alert",
            "Dangerous situation detected - warning hazard risk",
            "Safety concern identified - alert dangerous activity"
        ]
        
        results = {
            'model_name': model_name,
            'video_file': os.path.basename(video_path),
            'model_healthy': True,
            'configuration': config or {},
            'caption_results': []
        }
        
        total_loss = 0.0
        all_top_tokens = []
        all_top_probs = []
        
        for caption_idx, caption in enumerate(test_captions):
            inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=64).to("cuda")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    pixel_values=video_tensor.unsqueeze(0),
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
                
                # Get logits and compute loss
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    inputs.input_ids.view(-1),
                    ignore_index=-100
                )
                
                # Check if model produces finite outputs
                logits_finite = torch.isfinite(logits).all()
                loss_finite = torch.isfinite(loss).all()
                
                if not (logits_finite and loss_finite):
                    results['model_healthy'] = False
                    continue
                
                # Get probability distribution over vocabulary
                probs = F.softmax(logits[0, -1, :], dim=-1)  # Last token probabilities
                top_k_probs, top_k_indices = torch.topk(probs, k=15)
                
                # Decode top tokens
                top_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
                top_probs_list = [prob.item() for prob in top_k_probs]
                
                # Analyze danger keywords
                danger_analysis = analyze_danger_keywords(tokenizer, top_tokens, top_probs_list)
                
                caption_result = {
                    'caption': caption,
                    'loss': loss.item(),
                    'logits_min': logits.min().item(),
                    'logits_max': logits.max().item(),
                    'logits_std': logits.std().item(),
                    'top_tokens': top_tokens,
                    'top_probs': top_probs_list,
                    'danger_analysis': danger_analysis
                }
                
                results['caption_results'].append(caption_result)
                total_loss += loss.item()
                all_top_tokens.extend(top_tokens[:5])  # Top 5 from each caption
                all_top_probs.extend(top_probs_list[:5])
        
        # Aggregate results
        if results['caption_results']:
            results['avg_loss'] = total_loss / len(results['caption_results'])
            results['top_tokens_aggregate'] = list(set(all_top_tokens))  # Unique tokens
            results['overall_danger_analysis'] = analyze_danger_keywords(tokenizer, all_top_tokens, all_top_probs)
            
            print(f"    📊 {model_name}: Avg Loss={results['avg_loss']:.4f}, Healthy={results['model_healthy']}")
            print(f"    🔤 Unique top tokens: {len(results['top_tokens_aggregate'])}")
            print(f"    ⚠️  Danger keywords: {results['overall_danger_analysis']['danger_token_count']}")
        
        # Clean up
        del video_tensor
        return results
        
    except Exception as e:
        error_result = {
            'model_name': model_name,
            'video_file': os.path.basename(video_path),
            'error': str(e),
            'model_healthy': False
        }
        print(f"    ❌ {model_name}: Error - {str(e)[:50]}")
        return error_result

def load_enhanced_checkpoint(model, checkpoint_path):
    """Load enhanced LoRA checkpoint if available"""
    try:
        if not Path(checkpoint_path).exists():
            print(f"    ⚠️  Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        lora_state = checkpoint.get('lora_state_dict', {})
        
        loaded_params = 0
        for name, param in model.named_parameters():
            if name in lora_state:
                param.data.copy_(lora_state[name])
                loaded_params += 1
        
        print(f"    ✅ Loaded {loaded_params} LoRA parameters from checkpoint")
        print(f"    📊 Checkpoint info: Epoch {checkpoint.get('epoch', 'Unknown')}, Step {checkpoint.get('step', 'Unknown')}")
        return True
        
    except Exception as e:
        print(f"    ❌ Failed to load checkpoint: {str(e)[:50]}")
        return False

def compare_training_effectiveness(baseline_results, trained_results):
    """Comprehensive comparison of training effectiveness"""
    
    print("\n3️⃣ COMPREHENSIVE TRAINING EFFECTIVENESS ANALYSIS:")
    print("="*80)
    
    effectiveness_metrics = {
        'videos_compared': 0,
        'loss_improvements': [],
        'token_diversity_changes': [],
        'danger_keyword_improvements': [],
        'behavioral_changes': 0,
        'overall_effectiveness_score': 0.0
    }
    
    for i, (baseline, trained) in enumerate(zip(baseline_results, trained_results)):
        video_name = baseline.get('video_file', f'video_{i+1}')
        print(f"\n📹 Video {i+1}: {video_name}")
        
        if not (baseline.get('model_healthy', False) and trained.get('model_healthy', False)):
            print(f"   ⚠️  Skipping - model health issues")
            continue
        
        effectiveness_metrics['videos_compared'] += 1
        
        # Loss comparison
        baseline_loss = baseline.get('avg_loss', float('inf'))
        trained_loss = trained.get('avg_loss', float('inf'))
        loss_improvement = baseline_loss - trained_loss
        effectiveness_metrics['loss_improvements'].append(loss_improvement)
        
        print(f"   📊 Loss: Baseline={baseline_loss:.4f} → Trained={trained_loss:.4f} (Δ{loss_improvement:+.4f})")
        
        # Token diversity comparison
        baseline_tokens = set(baseline.get('top_tokens_aggregate', []))
        trained_tokens = set(trained.get('top_tokens_aggregate', []))
        
        new_tokens = trained_tokens - baseline_tokens
        lost_tokens = baseline_tokens - trained_tokens
        token_diversity_change = len(new_tokens) - len(lost_tokens)
        effectiveness_metrics['token_diversity_changes'].append(token_diversity_change)
        
        print(f"   🔤 Tokens: +{len(new_tokens)} new, -{len(lost_tokens)} lost (net: {token_diversity_change:+d})")
        if new_tokens:
            print(f"      New tokens: {', '.join(list(new_tokens)[:5])}")
        
        # Danger keyword analysis
        baseline_danger = baseline.get('overall_danger_analysis', {})
        trained_danger = trained.get('overall_danger_analysis', {})
        
        baseline_danger_score = baseline_danger.get('danger_score', 0.0)
        trained_danger_score = trained_danger.get('danger_score', 0.0)
        danger_improvement = trained_danger_score - baseline_danger_score
        effectiveness_metrics['danger_keyword_improvements'].append(danger_improvement)
        
        print(f"   ⚠️  Danger keywords: {baseline_danger.get('danger_token_count', 0)} → {trained_danger.get('danger_token_count', 0)}")
        print(f"      Danger score: {baseline_danger_score:.4f} → {trained_danger_score:.4f} (Δ{danger_improvement:+.4f})")
        
        # Overall behavioral change detection
        behavioral_change = (
            abs(loss_improvement) > 0.1 or  # Significant loss change
            len(new_tokens) > 2 or          # New token predictions
            danger_improvement > 0.01       # Improved danger detection
        )
        
        if behavioral_change:
            effectiveness_metrics['behavioral_changes'] += 1
            print(f"   ✅ Behavioral change detected!")
        else:
            print(f"   ❌ No significant behavioral change")
    
    # Calculate overall effectiveness
    if effectiveness_metrics['videos_compared'] > 0:
        avg_loss_improvement = sum(effectiveness_metrics['loss_improvements']) / len(effectiveness_metrics['loss_improvements'])
        avg_token_diversity = sum(effectiveness_metrics['token_diversity_changes']) / len(effectiveness_metrics['token_diversity_changes'])
        avg_danger_improvement = sum(effectiveness_metrics['danger_keyword_improvements']) / len(effectiveness_metrics['danger_keyword_improvements'])
        behavioral_change_rate = effectiveness_metrics['behavioral_changes'] / effectiveness_metrics['videos_compared']
        
        # Composite effectiveness score (0-100)
        effectiveness_score = (
            min(avg_loss_improvement * 10, 30) +      # Loss improvement (up to 30 points)
            min(avg_token_diversity * 2, 20) +        # Token diversity (up to 20 points)
            min(avg_danger_improvement * 100, 25) +   # Danger improvement (up to 25 points)
            behavioral_change_rate * 25               # Behavioral changes (up to 25 points)
        )
        effectiveness_metrics['overall_effectiveness_score'] = max(0, effectiveness_score)
        
        print(f"\n📊 OVERALL TRAINING EFFECTIVENESS:")
        print(f"   📈 Videos successfully compared: {effectiveness_metrics['videos_compared']}")
        print(f"   📉 Average loss improvement: {avg_loss_improvement:.4f}")
        print(f"   🔤 Average token diversity change: {avg_token_diversity:+.1f}")
        print(f"   ⚠️  Average danger score improvement: {avg_danger_improvement:+.4f}")
        print(f"   🔄 Behavioral change rate: {behavioral_change_rate:.1%}")
        print(f"   🎯 Overall effectiveness score: {effectiveness_score:.1f}/100")
        
        # Effectiveness interpretation
        if effectiveness_score >= 70:
            print(f"\n🏆 OUTSTANDING! Training was highly effective!")
        elif effectiveness_score >= 50:
            print(f"\n🎉 EXCELLENT! Training showed strong effectiveness!")
        elif effectiveness_score >= 30:
            print(f"\n✅ GOOD! Training showed meaningful improvements!")
        elif effectiveness_score >= 10:
            print(f"\n⚠️  PARTIAL: Training showed some improvements!")
        else:
            print(f"\n❌ MINIMAL: Training showed little effectiveness!")
            print(f"   💡 Suggestions: Increase learning rate, more epochs, larger LoRA rank")
    
    return effectiveness_metrics

def evaluate_training_effectiveness(dataset_dir, max_samples=5, checkpoint_path=None):
    """Main evaluation function for training effectiveness"""
    
    set_env()
    cleanup_memory()
    
    print("🚀 COMPREHENSIVE TRAINING EFFECTIVENESS EVALUATION")
    print("💡 Analyzing baseline vs enhanced trained model")
    
    # Find videos
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
                if len(video_files) >= max_samples:
                    break
        if len(video_files) >= max_samples:
            break
    
    if not video_files:
        print("❌ No videos found!")
        return
    
    print(f"📊 Testing {len(video_files)} videos:")
    for i, vf in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(vf)}")
    
    # Test baseline model
    print("\n1️⃣ BASELINE MODEL EVALUATION:")
    print("-" * 60)
    disable_torch_init()
    baseline_model, processor, tokenizer = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir=os.environ["HF_HOME"]
    )
    baseline_model.to("cuda")
    baseline_model.config.use_cache = False
    baseline_model.eval()
    
    print("✅ Baseline model loaded")
    
    baseline_results = []
    for video_path in video_files:
        result = comprehensive_model_test(baseline_model, processor, tokenizer, video_path, "baseline")
        baseline_results.append(result)
        cleanup_memory()
    
    # Clean up baseline model
    del baseline_model
    cleanup_memory()
    
    print("\n" + "="*80)
    print("⏳ Loading enhanced trained model...")
    
    # Test enhanced trained model
    print("\n2️⃣ ENHANCED TRAINED MODEL EVALUATION:")
    print("-" * 60)
    trained_model, _, _ = model_init(
        "DAMO-NLP-SG/VideoLLaMA2-7B-16F",
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir=os.environ["HF_HOME"]
    )
    trained_model.to("cuda")
    trained_model.config.use_cache = False
    
    # Enhanced LoRA configuration (same as training)
    enhanced_config = {
        'r': 8,
        'lora_alpha': 16,
        'target_modules': ["lm_head", "embed_tokens"],
        'lora_dropout': 0.1
    }
    
    config = LoraConfig(
        r=enhanced_config['r'],
        lora_alpha=enhanced_config['lora_alpha'],
        target_modules=enhanced_config['target_modules'],
        bias="none", 
        lora_dropout=enhanced_config['lora_dropout'], 
        task_type="CAUSAL_LM"
    )
    trained_model = get_peft_model(trained_model, config)
    convert_lora_to_fp32(trained_model)
    
    # Load checkpoint if provided
    if checkpoint_path:
        load_enhanced_checkpoint(trained_model, checkpoint_path)
    
    trained_model.eval()
    
    print("✅ Enhanced trained model loaded with LoRA")
    
    trained_results = []
    for video_path in video_files:
        result = comprehensive_model_test(trained_model, processor, tokenizer, video_path, "enhanced_trained", enhanced_config)
        trained_results.append(result)
        cleanup_memory()
    
    # Clean up trained model
    del trained_model, processor, tokenizer
    cleanup_memory()
    
    # Comprehensive comparison
    effectiveness_metrics = compare_training_effectiveness(baseline_results, trained_results)
    
    # Save comprehensive results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user': 'nofilsiddiqui-2000',
        'date': '2025-07-01',
        'evaluation_type': 'comprehensive_training_effectiveness',
        'videos_tested': len(video_files),
        'checkpoint_used': checkpoint_path,
        'baseline_results': baseline_results,
        'trained_results': trained_results,
        'effectiveness_metrics': effectiveness_metrics,
        'enhanced_config': enhanced_config
    }
    
    results_file = f"training_effectiveness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Comprehensive evaluation results saved to: {results_file}")
    print("🏁 TRAINING EFFECTIVENESS EVALUATION COMPLETE!")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced VBAD training effectiveness")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing test videos")
    parser.add_argument("--max-samples", type=int, default=5, help="Maximum videos to test")
    parser.add_argument("--checkpoint", help="Path to enhanced LoRA checkpoint file")
    args = parser.parse_args()
    
    evaluate_training_effectiveness(args.dataset_dir, args.max_samples, args.checkpoint)

if __name__ == "__main__":
    main()