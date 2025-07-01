#!/usr/bin/env python3
# TRAINING COMPARISON UTILITY - Compare conservative vs enhanced training approaches

import os, sys, json, argparse, subprocess
from pathlib import Path
from datetime import datetime

def run_command(command, description, timeout=3600):
    """Run a command and capture output"""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print("✅ Command completed successfully")
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {
                'success': False,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        return {'success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"❌ Command execution failed: {str(e)}")
        return {'success': False, 'error': str(e)}

def extract_training_metrics(result_file):
    """Extract key metrics from training results JSON"""
    try:
        if not Path(result_file).exists():
            return None
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        metrics = {
            'successful_steps': results.get('results', {}).get('successful_steps', 0),
            'success_rate': results.get('results', {}).get('success_rate', 0.0),
            'avg_loss': results.get('results', {}).get('avg_loss', 0.0),
            'training_effectiveness': results.get('results', {}).get('training_effectiveness', False),
            'model_health': results.get('results', {}).get('model_health', False),
            'loss_improvement': results.get('results', {}).get('loss_improvement', 0.0),
            'configuration': results.get('configuration', {}),
            'timestamp': results.get('timestamp', '')
        }
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error extracting metrics from {result_file}: {str(e)}")
        return None

def compare_approaches(dataset_dir, max_samples=10, timeout=1800):
    """Compare conservative vs enhanced training approaches"""
    
    print("🎯 VBAD TRAINING APPROACH COMPARISON")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Timeout per run: {timeout} seconds")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"comparison_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    comparison_results = {
        'timestamp': timestamp,
        'dataset_dir': dataset_dir,
        'max_samples': max_samples,
        'conservative_results': None,
        'enhanced_results': None,
        'comparison': None
    }
    
    # 1. Run Conservative Training (using existing vbad_simple_working.py)
    print(f"\n1️⃣ CONSERVATIVE TRAINING (vbad_simple_working.py)")
    conservative_cmd = f"python vbad_simple_working.py --dataset-dir {dataset_dir} --max-samples {max_samples}"
    conservative_result = run_command(conservative_cmd, "Running conservative training", timeout)
    
    # Look for conservative results file
    conservative_metrics = None
    if conservative_result['success']:
        # vbad_simple_working.py creates vbad_results.json
        conservative_metrics = extract_training_metrics("vbad_results.json")
        if conservative_metrics:
            # Move results file to comparison directory
            subprocess.run(f"mv vbad_results.json {results_dir}/conservative_results.json", shell=True)
    
    comparison_results['conservative_results'] = conservative_metrics
    
    # 2. Run Enhanced Training
    print(f"\n2️⃣ ENHANCED TRAINING (vbad_enhanced_training.py)")
    enhanced_cmd = f"python vbad_enhanced_training.py --dataset-dir {dataset_dir} --max-samples {max_samples} --learning-rate 1e-4 --lora-rank 8 --lora-alpha 16 --epochs 3"
    enhanced_result = run_command(enhanced_cmd, "Running enhanced training", timeout)
    
    # Look for enhanced results file
    enhanced_metrics = None
    if enhanced_result['success']:
        # Find the most recent enhanced results file
        enhanced_files = list(Path(".").glob("vbad_enhanced_results_*.json"))
        if enhanced_files:
            latest_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
            enhanced_metrics = extract_training_metrics(latest_file)
            if enhanced_metrics:
                # Move results file to comparison directory
                subprocess.run(f"mv {latest_file} {results_dir}/enhanced_results.json", shell=True)
    
    comparison_results['enhanced_results'] = enhanced_metrics
    
    # 3. Compare Results
    print(f"\n3️⃣ TRAINING COMPARISON ANALYSIS")
    print("="*70)
    
    if conservative_metrics and enhanced_metrics:
        comparison = analyze_training_comparison(conservative_metrics, enhanced_metrics)
        comparison_results['comparison'] = comparison
        print_comparison_results(comparison)
        
        # Save comprehensive comparison
        comparison_file = results_dir / "training_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"\n💾 Comparison results saved to: {comparison_file}")
        
        # Generate recommendation
        recommendation = generate_recommendation(comparison)
        print(f"\n🎯 RECOMMENDATION:")
        print(recommendation)
        
        return comparison_results
        
    else:
        print("❌ Unable to complete comparison - missing results")
        if not conservative_metrics:
            print("   Conservative training failed or no results found")
        if not enhanced_metrics:
            print("   Enhanced training failed or no results found")
        
        return comparison_results

def analyze_training_comparison(conservative, enhanced):
    """Analyze differences between conservative and enhanced training"""
    
    comparison = {
        'success_rate_improvement': enhanced['success_rate'] - conservative['success_rate'],
        'loss_improvement': conservative['avg_loss'] - enhanced['avg_loss'],
        'effectiveness_gained': enhanced['training_effectiveness'] and not conservative['training_effectiveness'],
        'model_health_maintained': enhanced['model_health'] and conservative['model_health'],
        'configuration_differences': {},
        'overall_improvement_score': 0.0
    }
    
    # Configuration differences
    cons_config = conservative.get('configuration', {})
    enh_config = enhanced.get('configuration', {})
    
    comparison['configuration_differences'] = {
        'learning_rate_ratio': enh_config.get('learning_rate', 1e-4) / cons_config.get('learning_rate', 1e-5) if cons_config.get('learning_rate', 1e-5) > 0 else 0,
        'lora_rank_ratio': enh_config.get('lora_rank', 8) / cons_config.get('lora_rank', 4) if cons_config.get('lora_rank', 4) > 0 else 0,
        'epochs_ratio': enh_config.get('epochs', 3) / cons_config.get('epochs', 1) if cons_config.get('epochs', 1) > 0 else 0,
    }
    
    # Calculate overall improvement score (0-100)
    score = 0
    
    # Success rate improvement (0-30 points)
    score += min(comparison['success_rate_improvement'] * 100, 30)
    
    # Loss improvement (0-25 points) 
    if comparison['loss_improvement'] > 0:
        score += min(comparison['loss_improvement'] * 5, 25)
    
    # Training effectiveness (0-25 points)
    if comparison['effectiveness_gained']:
        score += 25
    
    # Model health maintained (0-20 points)
    if comparison['model_health_maintained']:
        score += 20
    
    comparison['overall_improvement_score'] = max(0, score)
    
    return comparison

def print_comparison_results(comparison):
    """Print formatted comparison results"""
    
    print(f"📊 Success Rate: {comparison['success_rate_improvement']:+.1%}")
    print(f"📉 Loss Improvement: {comparison['loss_improvement']:+.4f}")
    print(f"⚡ Training Effectiveness: {'✅ Gained' if comparison['effectiveness_gained'] else '❌ Not gained'}")
    print(f"🏥 Model Health: {'✅ Maintained' if comparison['model_health_maintained'] else '❌ Lost'}")
    
    config_diff = comparison['configuration_differences']
    print(f"\n📋 Configuration Changes:")
    print(f"   Learning Rate: {config_diff.get('learning_rate_ratio', 0):.1f}x increase")
    print(f"   LoRA Rank: {config_diff.get('lora_rank_ratio', 0):.1f}x increase")
    print(f"   Epochs: {config_diff.get('epochs_ratio', 0):.1f}x increase")
    
    score = comparison['overall_improvement_score']
    print(f"\n🎯 Overall Improvement Score: {score:.1f}/100")
    
    if score >= 70:
        print("🏆 OUTSTANDING improvement with enhanced training!")
    elif score >= 50:
        print("🎉 EXCELLENT improvement with enhanced training!")
    elif score >= 30:
        print("✅ GOOD improvement with enhanced training!")
    elif score >= 10:
        print("⚠️  MODEST improvement with enhanced training")
    else:
        print("❌ MINIMAL improvement - consider further adjustments")

def generate_recommendation(comparison):
    """Generate recommendations based on comparison results"""
    
    score = comparison['overall_improvement_score']
    
    if score >= 70:
        return (
            "🎯 Enhanced training is highly recommended!\n"
            "   • Significant improvements in all key metrics\n"
            "   • Safe to deploy enhanced settings for production training\n"
            "   • Consider fine-tuning hyperparameters for even better results"
        )
    elif score >= 50:
        return (
            "✅ Enhanced training shows strong benefits!\n"
            "   • Good improvements in most metrics\n" 
            "   • Recommended for regular use\n"
            "   • Monitor model health during longer training runs"
        )
    elif score >= 30:
        return (
            "⚠️  Enhanced training shows modest benefits\n"
            "   • Some improvements but results mixed\n"
            "   • Test with larger datasets or different hyperparameters\n"
            "   • Consider gradual transition from conservative settings"
        )
    else:
        return (
            "❌ Enhanced training needs adjustment\n"
            "   • Minimal improvements over conservative approach\n"
            "   • Reduce learning rate or increase regularization\n"
            "   • Debug potential issues with dataset or configuration\n"
            "   • Stick with conservative approach until issues resolved"
        )

def main():
    parser = argparse.ArgumentParser(description="Compare conservative vs enhanced VBAD training approaches")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing video files")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum videos to test with")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per training run (seconds)")
    
    args = parser.parse_args()
    
    # Verify scripts exist
    if not Path("vbad_simple_working.py").exists():
        print("❌ vbad_simple_working.py not found!")
        return
    
    if not Path("vbad_enhanced_training.py").exists():
        print("❌ vbad_enhanced_training.py not found!")
        return
    
    if not Path(args.dataset_dir).exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return
    
    # Run comparison
    results = compare_approaches(args.dataset_dir, args.max_samples, args.timeout)
    
    if results and results.get('comparison'):
        print(f"\n🏁 Training comparison completed successfully!")
    else:
        print(f"\n⚠️  Training comparison completed with issues")

if __name__ == "__main__":
    main()