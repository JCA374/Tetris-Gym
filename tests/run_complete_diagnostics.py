#!/usr/bin/env python3
"""
Complete Tetris AI Diagnostic Suite
Run this to check if the model can see the board properly and identify training issues
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_diagnostic_sequence():
    """Run the complete diagnostic sequence to identify training issues"""
    
    print("🔍 TETRIS AI COMPLETE DIAGNOSTIC SUITE")
    print("=" * 60)
    print("This will run all diagnostic tools to identify why training is stuck")
    print()
    
    # Check if we have the necessary files
    diagnostics_to_run = [
        ("visual_board_check.py", "Quick visual check of agent vision"),
        ("tetris_vision_diagnostic.py", "Complete vision system analysis"),  
        ("tetris_diagnostic.py", "Professional training analysis"),
        ("config.py", "Test corrected environment setup")
    ]
    
    missing_files = []
    for filename, description in diagnostics_to_run:
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        print("❌ Missing diagnostic files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all diagnostic files are in the current directory.")
        return False
    
    print("✅ All diagnostic files found")
    print()
    
    # Run diagnostic sequence
    results = {}
    
    # 1. Quick visual check
    print("🎯 STEP 1: Visual Board Check")
    print("-" * 40)
    try:
        result = subprocess.run([sys.executable, "visual_board_check.py"], 
                              capture_output=True, text=True, timeout=120)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        results['visual_check'] = result.returncode == 0
        print(f"Result: {'✅ PASSED' if results['visual_check'] else '❌ FAILED'}")
    except Exception as e:
        print(f"❌ Failed to run visual check: {e}")
        results['visual_check'] = False
    
    print("\n" + "="*60 + "\n")
    
    # 2. Complete vision diagnostic
    print("🔬 STEP 2: Complete Vision Analysis")
    print("-" * 40)
    try:
        result = subprocess.run([sys.executable, "tetris_vision_diagnostic.py"], 
                              capture_output=True, text=True, timeout=180)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        results['vision_analysis'] = result.returncode == 0
        print(f"Result: {'✅ PASSED' if results['vision_analysis'] else '❌ FAILED'}")
    except Exception as e:
        print(f"❌ Failed to run vision analysis: {e}")
        results['vision_analysis'] = False
    
    print("\n" + "="*60 + "\n")
    
    # 3. Test environment setup
    print("⚙️ STEP 3: Environment Setup Test")
    print("-" * 40)
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "from config import corrected_vision_test; corrected_vision_test()"], 
                              capture_output=True, text=True, timeout=120)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        results['environment_test'] = result.returncode == 0
        print(f"Result: {'✅ PASSED' if results['environment_test'] else '❌ FAILED'}")
    except Exception as e:
        print(f"❌ Failed to run environment test: {e}")
        results['environment_test'] = False
    
    print("\n" + "="*60 + "\n")
    
    # 4. Training diagnostic (shorter version)
    print("📊 STEP 4: Training Analysis (10 episodes)")
    print("-" * 40)
    try:
        result = subprocess.run([sys.executable, "tetris_diagnostic.py", 
                               "--episodes", "10", "--analyze-board", "--analyze-actions", 
                               "--save-plots", "--output-dir", "diagnostic_results"],
                              capture_output=True, text=True, timeout=300)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        results['training_analysis'] = result.returncode == 0
        print(f"Result: {'✅ PASSED' if results['training_analysis'] else '❌ FAILED'}")
    except Exception as e:
        print(f"❌ Failed to run training analysis: {e}")
        results['training_analysis'] = False
    
    print("\n" + "="*80 + "\n")
    
    # Summary and recommendations
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print()
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("\n" + "="*60)
    
    # Specific recommendations based on results
    if not results.get('visual_check', False):
        print("🚨 CRITICAL: Visual board check failed!")
        print("   The model cannot see the Tetris board correctly.")
        print("   📋 Action: Fix TetrisBoardWrapper or TetrisObservationWrapper")
        print("   📋 Check: Board dimensions, aspect ratio, spatial structure")
        print()
    
    if not results.get('vision_analysis', False):
        print("🚨 CRITICAL: Vision analysis failed!")
        print("   Spatial processing or board detection issues.")
        print("   📋 Action: Review preprocessing pipeline")
        print("   📋 Check: Board extraction from dict observations")
        print()
    
    if not results.get('environment_test', False):
        print("🚨 CRITICAL: Environment setup failed!")
        print("   Configuration issues with 24×18 board handling.")
        print("   📋 Action: Fix config.py board wrappers")
        print("   📋 Check: CorrectedTetrisBoardWrapper implementation")
        print()
    
    if not results.get('training_analysis', False):
        print("⚠️  WARNING: Training analysis issues!")
        print("   May indicate agent or training loop problems.")
        print("   📋 Action: Check agent.py and training pipeline")
        print()
    
    # Overall recommendations
    if passed_tests == total_tests:
        print("🎉 ALL DIAGNOSTICS PASSED!")
        print("\nIf training is still stuck, the issue is likely:")
        print("   1. Hyperparameters (epsilon, learning rate, reward shaping)")
        print("   2. Network architecture not suitable for Tetris")
        print("   3. Training loop logic issues")
        print("   4. Need for curriculum learning or action masking")
        print("\n📋 Next steps:")
        print("   • Run emergency epsilon fix: python emergency_epsilon_fix.py")
        print("   • Try plateau breaker: python break_plateau_train.py")
        print("   • Check recent training logs for epsilon values")
        
    elif passed_tests >= total_tests // 2:
        print("⚠️  PARTIAL SUCCESS - Some vision issues remain")
        print("\n📋 Priority fixes needed:")
        print("   1. Fix failing diagnostic tests above")
        print("   2. Verify board preprocessing works correctly")
        print("   3. Test with simple environment first")
        
    else:
        print("🚨 CRITICAL FAILURE - Major vision system issues!")
        print("\n📋 Emergency actions:")
        print("   1. Review config.py TetrisBoardWrapper implementation")
        print("   2. Test with direct feature mode (no CNN)")
        print("   3. Verify Tetris Gymnasium installation")
        print("   4. Check board dimension assumptions (24×18 vs 20×10)")
    
    print("\n📁 Check these files for detailed results:")
    print("   • agent_vision_check.png (if visual check ran)")
    print("   • diagnostic_results/ directory (if training analysis ran)")
    print("   • Any error logs above")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    print("Starting complete diagnostic sequence...")
    print("This may take 5-10 minutes to complete all tests.")
    print()
    
    success = run_diagnostic_sequence()
    
    print("\n" + "="*80)
    if success:
        print("🎯 DIAGNOSTICS COMPLETE - SYSTEM HEALTHY")
        print("If still stuck, issue is in training hyperparameters or logic.")
    else:
        print("🔧 DIAGNOSTICS COMPLETE - FIXES NEEDED") 
        print("Address the failing tests above before continuing training.")
    print("="*80)