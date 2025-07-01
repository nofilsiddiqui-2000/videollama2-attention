#!/usr/bin/env python3
# CONFIG LOADER UTILITY - Helper for loading and using training configurations

import json
import argparse
from pathlib import Path

def load_config(config_name="enhanced", config_file="config/training_configs.json"):
    """Load a specific training configuration"""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_file}")
            return None
        
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        if config_name not in configs.get('training_configs', {}):
            print(f"❌ Config '{config_name}' not found!")
            print(f"Available configs: {list(configs.get('training_configs', {}).keys())}")
            return None
        
        config = configs['training_configs'][config_name]
        print(f"✅ Loaded '{config_name}' configuration")
        print(f"   Description: {config.get('description', 'No description')}")
        return config
        
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return None

def generate_command(config, script="vbad_enhanced_training.py", dataset_dir=None, max_samples=15):
    """Generate command line from configuration"""
    if not config:
        return None
    
    cmd_parts = [f"python {script}"]
    
    if dataset_dir:
        cmd_parts.append(f"--dataset-dir {dataset_dir}")
    else:
        cmd_parts.append("--dataset-dir /path/to/videos")
    
    cmd_parts.append(f"--learning-rate {config.get('learning_rate', 1e-4)}")
    cmd_parts.append(f"--lora-rank {config.get('lora_rank', 8)}")
    cmd_parts.append(f"--lora-alpha {config.get('lora_alpha', 16)}")
    cmd_parts.append(f"--epochs {config.get('epochs', 3)}")
    cmd_parts.append(f"--gradient-clip {config.get('gradient_clip', 1.0)}")
    cmd_parts.append(f"--param-clamp {config.get('param_clamp', 5.0)}")
    cmd_parts.append(f"--max-samples {max_samples}")
    
    return " \\\n    ".join(cmd_parts)

def print_config_details(config):
    """Print detailed configuration information"""
    if not config:
        return
    
    print(f"\n📋 Configuration Details:")
    print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"   LoRA Rank: {config.get('lora_rank', 'N/A')}")
    print(f"   LoRA Alpha: {config.get('lora_alpha', 'N/A')}")
    print(f"   Target Modules: {config.get('target_modules', 'N/A')}")
    print(f"   Epochs: {config.get('epochs', 'N/A')}")
    print(f"   Gradient Clip: {config.get('gradient_clip', 'N/A')}")
    print(f"   Parameter Clamp: {config.get('param_clamp', 'N/A')}")
    print(f"   Captions: {len(config.get('captions', []))} captions")
    print(f"   Use Case: {config.get('use_case', 'N/A')}")

def list_available_configs(config_file="config/training_configs.json"):
    """List all available configurations"""
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        training_configs = configs.get('training_configs', {})
        eval_configs = configs.get('evaluation_configs', {})
        
        print("🎯 Available Training Configurations:")
        for name, config in training_configs.items():
            print(f"   • {name}: {config.get('description', 'No description')}")
        
        print(f"\n📊 Available Evaluation Configurations:")
        for name, config in eval_configs.items():
            print(f"   • {name}: {config.get('description', 'No description')}")
            
    except Exception as e:
        print(f"❌ Error listing configs: {str(e)}")

def compare_configs(config1_name, config2_name, config_file="config/training_configs.json"):
    """Compare two configurations side by side"""
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        training_configs = configs.get('training_configs', {})
        
        if config1_name not in training_configs or config2_name not in training_configs:
            print(f"❌ One or both configs not found!")
            return
        
        config1 = training_configs[config1_name]
        config2 = training_configs[config2_name]
        
        print(f"📊 Comparing '{config1_name}' vs '{config2_name}':")
        print(f"{'Parameter':<20} {'|':<1} {config1_name:<15} {'|':<1} {config2_name:<15}")
        print("-" * 55)
        
        params = ['learning_rate', 'lora_rank', 'lora_alpha', 'epochs', 'gradient_clip', 'param_clamp']
        for param in params:
            val1 = config1.get(param, 'N/A')
            val2 = config2.get(param, 'N/A')
            print(f"{param:<20} | {str(val1):<15} | {str(val2):<15}")
        
        # Special handling for arrays
        modules1 = config1.get('target_modules', [])
        modules2 = config2.get('target_modules', [])
        print(f"{'target_modules':<20} | {str(len(modules1)) + ' modules':<15} | {str(len(modules2)) + ' modules':<15}")
        
        captions1 = config1.get('captions', [])
        captions2 = config2.get('captions', [])
        print(f"{'captions':<20} | {str(len(captions1)) + ' captions':<15} | {str(len(captions2)) + ' captions':<15}")
        
    except Exception as e:
        print(f"❌ Error comparing configs: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Configuration loader utility for VBAD enhanced training")
    parser.add_argument("--list", action="store_true", help="List all available configurations")
    parser.add_argument("--load", type=str, help="Load and display specific configuration")
    parser.add_argument("--command", type=str, help="Generate command for specific configuration")
    parser.add_argument("--compare", nargs=2, help="Compare two configurations")
    parser.add_argument("--dataset-dir", type=str, help="Dataset directory for command generation")
    parser.add_argument("--max-samples", type=int, default=15, help="Max samples for command generation")
    parser.add_argument("--config-file", default="config/training_configs.json", help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_configs(args.config_file)
    
    elif args.load:
        config = load_config(args.load, args.config_file)
        if config:
            print_config_details(config)
    
    elif args.command:
        config = load_config(args.command, args.config_file)
        if config:
            print_config_details(config)
            command = generate_command(config, dataset_dir=args.dataset_dir, max_samples=args.max_samples)
            print(f"\n💻 Generated Command:")
            print(command)
    
    elif args.compare:
        compare_configs(args.compare[0], args.compare[1], args.config_file)
    
    else:
        print("🚀 VBAD Configuration Loader Utility")
        print("\nUsage examples:")
        print("  python config_loader.py --list")
        print("  python config_loader.py --load enhanced")
        print("  python config_loader.py --command enhanced --dataset-dir /path/to/videos")
        print("  python config_loader.py --compare conservative enhanced")

if __name__ == "__main__":
    main()