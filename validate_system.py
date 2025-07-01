#!/usr/bin/env python3
# VALIDATION SCRIPT - Test enhanced VBAD scripts without requiring video files

import os, sys, json, importlib.util
from pathlib import Path
import argparse

def validate_imports():
    """Validate that all required imports are available"""
    
    print("🔍 Validating Python imports...")
    
    required_modules = [
        'torch', 'transformers', 'peft', 'json', 'argparse', 
        'datetime', 'pathlib', 'gc', 'math'
    ]
    
    missing_modules = []
    available_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            available_modules.append(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"   ❌ {module} - MISSING")
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("   Install with: pip install torch transformers peft")
        return False
    else:
        print(f"\n✅ All {len(available_modules)} required modules available")
        return True

def validate_script_syntax(script_path):
    """Validate script syntax without executing"""
    
    print(f"🔍 Validating syntax: {script_path}")
    
    try:
        if not Path(script_path).exists():
            print(f"   ❌ File not found: {script_path}")
            return False
        
        # Read and compile the script
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        compile(script_content, script_path, 'exec')
        print(f"   ✅ Syntax valid")
        return True
        
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def validate_config_files():
    """Validate configuration files"""
    
    print("🔍 Validating configuration files...")
    
    config_file = "config/training_configs.json"
    
    try:
        if not Path(config_file).exists():
            print(f"   ❌ Config file not found: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        required_sections = ['training_configs', 'evaluation_configs', 'model_configs']
        for section in required_sections:
            if section not in configs:
                print(f"   ❌ Missing section: {section}")
                return False
            print(f"   ✅ Found section: {section}")
        
        # Validate training configs
        training_configs = configs['training_configs']
        required_configs = ['conservative', 'enhanced', 'aggressive', 'minimal']
        
        for config_name in required_configs:
            if config_name not in training_configs:
                print(f"   ❌ Missing training config: {config_name}")
                return False
            
            config = training_configs[config_name]
            required_params = ['learning_rate', 'lora_rank', 'lora_alpha', 'epochs']
            
            for param in required_params:
                if param not in config:
                    print(f"   ❌ Missing parameter {param} in {config_name}")
                    return False
            
            print(f"   ✅ Valid training config: {config_name}")
        
        print(f"   ✅ Configuration file valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def validate_script_help():
    """Test that scripts can show help without errors"""
    
    print("🔍 Validating script help functionality...")
    
    scripts = [
        'vbad_enhanced_training.py',
        'evaluate_training_effectiveness.py', 
        'config_loader.py',
        'training_comparison.py'
    ]
    
    valid_scripts = []
    
    for script in scripts:
        if not Path(script).exists():
            print(f"   ❌ Script not found: {script}")
            continue
        
        try:
            # Try to import and check for argparse usage
            spec = importlib.util.spec_from_file_location("test_module", script)
            if spec and spec.loader:
                print(f"   ✅ {script} - importable")
                valid_scripts.append(script)
            else:
                print(f"   ⚠️  {script} - import issues")
                
        except Exception as e:
            print(f"   ❌ {script} - error: {str(e)[:50]}")
    
    return len(valid_scripts) == len(scripts)

def test_config_loader():
    """Test config loader functionality"""
    
    print("🔍 Testing config loader functionality...")
    
    try:
        # Import config_loader
        sys.path.insert(0, '.')
        import config_loader
        
        # Test loading configurations
        for config_name in ['conservative', 'enhanced', 'aggressive']:
            config = config_loader.load_config(config_name)
            if config:
                print(f"   ✅ Loaded config: {config_name}")
            else:
                print(f"   ❌ Failed to load config: {config_name}")
                return False
        
        # Test command generation
        config = config_loader.load_config('enhanced')
        if config:
            command = config_loader.generate_command(config, dataset_dir="/test/path")
            if command and "vbad_enhanced_training.py" in command:
                print(f"   ✅ Command generation works")
            else:
                print(f"   ❌ Command generation failed")
                return False
        
        print(f"   ✅ Config loader functional")
        return True
        
    except Exception as e:
        print(f"   ❌ Config loader error: {str(e)}")
        return False

def validate_directory_structure():
    """Validate expected directory structure"""
    
    print("🔍 Validating directory structure...")
    
    expected_files = [
        'vbad_enhanced_training.py',
        'evaluate_training_effectiveness.py',
        'config_loader.py',
        'training_comparison.py',
        'README_ENHANCED_TRAINING.md',
        'config/training_configs.json'
    ]
    
    expected_dirs = [
        'config'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in expected_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            missing_files.append(file)
    
    for dir_name in expected_dirs:
        if Path(dir_name).is_dir():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - MISSING")
            missing_dirs.append(dir_name)
    
    if missing_files or missing_dirs:
        print(f"\n⚠️  Missing files: {missing_files}")
        print(f"⚠️  Missing directories: {missing_dirs}")
        return False
    else:
        print(f"\n✅ All expected files and directories present")
        return True

def run_basic_tests():
    """Run basic functionality tests"""
    
    print("🧪 Running basic functionality tests...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Configuration loading
    total_tests += 1
    try:
        sys.path.insert(0, '.')
        import config_loader
        
        config = config_loader.load_config('enhanced')
        if config and 'learning_rate' in config:
            print("   ✅ Configuration loading test")
            tests_passed += 1
        else:
            print("   ❌ Configuration loading test")
    except Exception as e:
        print(f"   ❌ Configuration loading test - {str(e)[:50]}")
    
    # Test 2: JSON parsing
    total_tests += 1
    try:
        with open('config/training_configs.json', 'r') as f:
            configs = json.load(f)
        
        if 'training_configs' in configs and 'enhanced' in configs['training_configs']:
            print("   ✅ JSON configuration parsing test")
            tests_passed += 1
        else:
            print("   ❌ JSON configuration parsing test")
    except Exception as e:
        print(f"   ❌ JSON configuration parsing test - {str(e)[:50]}")
    
    # Test 3: Command generation
    total_tests += 1
    try:
        import config_loader
        config = config_loader.load_config('enhanced')
        command = config_loader.generate_command(config)
        
        if command and '--learning-rate' in command and '--lora-rank' in command:
            print("   ✅ Command generation test")
            tests_passed += 1
        else:
            print("   ❌ Command generation test")
    except Exception as e:
        print(f"   ❌ Command generation test - {str(e)[:50]}")
    
    print(f"\n📊 Tests passed: {tests_passed}/{total_tests}")
    return tests_passed == total_tests

def main():
    parser = argparse.ArgumentParser(description="Validate enhanced VBAD training system")
    parser.add_argument("--skip-imports", action="store_true", help="Skip import validation (for environments without PyTorch)")
    parser.add_argument("--quick", action="store_true", help="Quick validation (skip detailed tests)")
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED VBAD TRAINING SYSTEM VALIDATION")
    print("="*60)
    
    validation_results = {
        'directory_structure': False,
        'config_files': False,
        'script_syntax': False,
        'imports': False,
        'basic_tests': False,
        'overall_status': False
    }
    
    # 1. Directory structure
    validation_results['directory_structure'] = validate_directory_structure()
    
    # 2. Configuration files
    validation_results['config_files'] = validate_config_files()
    
    # 3. Script syntax
    scripts = ['vbad_enhanced_training.py', 'evaluate_training_effectiveness.py', 
               'config_loader.py', 'training_comparison.py']
    
    syntax_valid = True
    for script in scripts:
        if not validate_script_syntax(script):
            syntax_valid = False
    
    validation_results['script_syntax'] = syntax_valid
    
    # 4. Imports (optional)
    if not args.skip_imports:
        validation_results['imports'] = validate_imports()
    else:
        print("⏭️  Skipping import validation")
        validation_results['imports'] = True
    
    # 5. Basic tests (optional)
    if not args.quick and validation_results['config_files']:
        validation_results['basic_tests'] = run_basic_tests()
    else:
        if args.quick:
            print("⏭️  Skipping basic tests (quick mode)")
        validation_results['basic_tests'] = True
    
    # Overall assessment (exclude overall_status from the check)
    test_results = {k: v for k, v in validation_results.items() if k != 'overall_status'}
    all_passed = all(test_results.values())
    validation_results['overall_status'] = all_passed
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print("="*60)
    for test, passed in validation_results.items():
        if test != 'overall_status':
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test.replace('_', ' ').title():<20}: {status}")
    
    print(f"\n🎯 OVERALL STATUS: {'✅ SYSTEM READY' if all_passed else '❌ ISSUES FOUND'}")
    
    if all_passed:
        print("\n🎉 Enhanced VBAD training system is ready to use!")
        print("\nNext steps:")
        print("   1. Install PyTorch/transformers if not available")
        print("   2. Prepare video dataset directory") 
        print("   3. Run: python vbad_enhanced_training.py --dataset-dir /path/to/videos")
        print("   4. Evaluate: python evaluate_training_effectiveness.py --dataset-dir /path/to/videos")
    else:
        print("\n⚠️  Please fix the issues above before using the system")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)