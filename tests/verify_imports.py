#!/usr/bin/env python3
"""
Quick script to verify all test files can import their dependencies.
Run this from the tests folder to verify everything works.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

print("="*70)
print("🔍 VERIFYING TEST FILE IMPORTS")
print("="*70)

tests = {
    "test_actions_simple.py": ["config", "numpy"],
    "test_environment_rendering.py": ["gymnasium", "numpy", "tetris_gymnasium"],
    "verify_training_actions.py": ["torch", "numpy", "config", "src.agent"],
    "diagnose_training.py": ["pathlib"],  # Minimal imports
    "diagnose_model.py": ["torch", "numpy", "config", "src.agent"],
    "test_reward_helpers.py": ["numpy", "src.reward_shaping"],
}

results = {}

for test_file, modules in tests.items():
    print(f"\n📄 {test_file}")
    print("-" * 70)

    success = True
    for module_name in modules:
        try:
            if "." in module_name:
                # Handle nested imports like src.agent
                parts = module_name.split(".")
                exec(f"from {parts[0]} import {parts[1]}")
            else:
                exec(f"import {module_name}")
            print(f"   ✅ {module_name}")
        except ImportError as e:
            print(f"   ❌ {module_name} - {str(e)[:50]}")
            success = False
        except Exception as e:
            print(f"   ⚠️  {module_name} - {str(e)[:50]}")
            success = False

    results[test_file] = success

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

for test_file, success in results.items():
    status = "✅" if success else "❌"
    print(f"{status} {test_file}")

all_success = all(results.values())
print("\n" + "="*70)
if all_success:
    print("✅ ALL IMPORTS SUCCESSFUL - All tests can run from tests folder!")
else:
    print("⚠️  SOME IMPORTS FAILED - Install missing dependencies:")
    print("   pip install numpy torch gymnasium tetris-gymnasium")
print("="*70)
