#!/usr/bin/env python3
"""
Fixed diagnostic script for tetris-gymnasium
"""

import sys
import os


def check_imports():
    """Check what we can import"""
    print("=== IMPORT CHECK ===")

    try:
        import gymnasium as gym
        print(f"✅ gymnasium: {gym.__version__}")
    except ImportError as e:
        print(f"❌ gymnasium: {e}")
        return False

    try:
        import tetris_gymnasium
        print(f"✅ tetris_gymnasium imported")
        print(f"   Path: {tetris_gymnasium.__file__}")
        print(
            f"   Dir contents: {os.listdir(os.path.dirname(tetris_gymnasium.__file__))}")
    except ImportError as e:
        print(f"❌ tetris_gymnasium: {e}")
        return False

    return True


def check_registrations():
    """Check what environments are registered (with fixed API)"""
    print("\n=== ENVIRONMENT REGISTRATION CHECK ===")

    import gymnasium as gym

    try:
        # Try different ways to access registry
        if hasattr(gym.envs, 'registry'):
            if hasattr(gym.envs.registry, 'env_specs'):
                registry = gym.envs.registry.env_specs
            elif hasattr(gym.envs.registry, 'all'):
                registry = gym.envs.registry.all()
            else:
                registry = gym.envs.registry
        else:
            print("❌ No registry found")
            return []

        # Find tetris environments
        tetris_envs = []

        if isinstance(registry, dict):
            for env_id in registry.keys():
                if 'tetris' in env_id.lower():
                    tetris_envs.append(env_id)
        else:
            # Try to iterate
            try:
                for env_spec in registry:
                    if hasattr(env_spec, 'id') and 'tetris' in env_spec.id.lower():
                        tetris_envs.append(env_spec.id)
            except:
                print("❌ Cannot iterate registry")

        if tetris_envs:
            print("Found Tetris environments:")
            for env_id in tetris_envs:
                print(f"  ✅ {env_id}")
        else:
            print("❌ No Tetris environments found in registry")

        return tetris_envs

    except Exception as e:
        print(f"❌ Registry check failed: {e}")
        return []


def explore_tetris_module():
    """Explore the tetris_gymnasium module structure"""
    print("\n=== TETRIS MODULE EXPLORATION ===")

    try:
        import tetris_gymnasium
        module_dir = os.path.dirname(tetris_gymnasium.__file__)

        print(f"Module directory: {module_dir}")

        # Walk through the directory structure
        for root, dirs, files in os.walk(module_dir):
            level = root.replace(module_dir, '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")

            subindent = '  ' * (level + 1)
            for file in files:
                if file.endswith('.py'):
                    print(f"{subindent}{file}")

        # Try to find environment classes
        possible_paths = [
            'tetris_gymnasium.envs.tetris',
            'tetris_gymnasium.envs.tetris_env',
            'tetris_gymnasium.tetris',
            'tetris_gymnasium.env',
        ]

        working_imports = []

        for path in possible_paths:
            try:
                module = __import__(path, fromlist=[''])
                print(f"✅ Can import: {path}")
                print(
                    f"   Contents: {[attr for attr in dir(module) if not attr.startswith('_')]}")
                working_imports.append(path)
            except Exception as e:
                print(f"❌ Cannot import {path}: {e}")

        return working_imports

    except Exception as e:
        print(f"❌ Module exploration failed: {e}")
        return []


def try_direct_environment_creation():
    """Try to create environment directly"""
    print("\n=== DIRECT ENVIRONMENT CREATION ===")

    # Possible class locations and names
    attempts = [
        ('tetris_gymnasium.envs.tetris', 'Tetris'),
        ('tetris_gymnasium.envs.tetris', 'TetrisEnv'),
        ('tetris_gymnasium.envs.tetris_env', 'Tetris'),
        ('tetris_gymnasium.envs.tetris_env', 'TetrisEnv'),
        ('tetris_gymnasium.tetris', 'Tetris'),
        ('tetris_gymnasium.env', 'Tetris'),
    ]

    working_class = None

    for module_path, class_name in attempts:
        try:
            print(f"Trying: from {module_path} import {class_name}")
            module = __import__(module_path, fromlist=[class_name])
            env_class = getattr(module, class_name)

            # Try to create an instance
            env = env_class()
            print(f"✅ SUCCESS: {module_path}.{class_name}")
            print(f"   Environment created: {type(env)}")

            # Check basic properties
            if hasattr(env, 'observation_space'):
                print(f"   Observation space: {env.observation_space}")
            if hasattr(env, 'action_space'):
                print(f"   Action space: {env.action_space}")

            working_class = (module_path, class_name)
            break

        except Exception as e:
            print(f"❌ {module_path}.{class_name}: {e}")

    return working_class


def try_manual_registration():
    """Try manual registration approaches"""
    print("\n=== MANUAL REGISTRATION TEST ===")

    import gymnasium as gym
    from gymnasium.envs.registration import register

    working_registrations = []

    # First, find a working class
    attempts = [
        ('tetris_gymnasium.envs.tetris:Tetris', 'TetrisManual-v0'),
        ('tetris_gymnasium.envs.tetris_env:Tetris', 'TetrisManual2-v0'),
        ('tetris_gymnasium.tetris:Tetris', 'TetrisManual3-v0'),
    ]

    for entry_point, env_id in attempts:
        try:
            print(f"Registering: {env_id} -> {entry_point}")

            register(
                id=env_id,
                entry_point=entry_point,
            )

            # Test it
            env = gym.make(env_id)
            print(f"✅ Registration works: {env_id}")
            env.close()
            working_registrations.append((entry_point, env_id))

        except Exception as e:
            print(f"❌ Registration failed for {env_id}: {e}")

    return working_registrations


def create_working_solution():
    """Create a working solution based on what we found"""
    print("\n=== SOLUTION GENERATION ===")

    # Run all diagnostics
    if not check_imports():
        return None

    tetris_envs = check_registrations()
    working_imports = explore_tetris_module()
    direct_class = try_direct_environment_creation()
    manual_regs = try_manual_registration()

    # Determine best solution
    if tetris_envs:
        print(f"✅ SOLUTION 1: Use existing registration")
        return ("existing", tetris_envs[0])

    elif manual_regs:
        print(f"✅ SOLUTION 2: Use manual registration")
        return ("manual", manual_regs[0])

    elif direct_class:
        print(f"✅ SOLUTION 3: Use direct class import")
        return ("direct", direct_class)

    else:
        print("❌ No working solution found")
        return None


def generate_config_code(solution):
    """Generate working config.py code"""
    if not solution:
        return None

    solution_type, value = solution

    print(f"\n🎉 GENERATING CONFIG FOR: {solution_type}")

    if solution_type == "existing":
        env_name = value
        config_code = f'''
# Working config.py for existing registration
ENV_NAME = "{env_name}"

def make_env(env_name=None, render_mode=None, **kwargs):
    import gymnasium as gym
    if env_name is None:
        env_name = ENV_NAME
    return gym.make(env_name, render_mode=render_mode or "rgb_array", **kwargs)
'''

    elif solution_type == "manual":
        entry_point, env_id = value
        config_code = f'''
# Working config.py with manual registration
import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment
register(
    id="{env_id}",
    entry_point="{entry_point}",
)

ENV_NAME = "{env_id}"

def make_env(env_name=None, render_mode=None, **kwargs):
    if env_name is None:
        env_name = ENV_NAME
    return gym.make(env_name, render_mode=render_mode or "rgb_array", **kwargs)
'''

    elif solution_type == "direct":
        module_path, class_name = value
        config_code = f'''
# Working config.py with direct import
from {module_path} import {class_name}
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

ENV_NAME = "Direct-{class_name}"

class DirectWrapper:
    def __init__(self, render_mode=None, **kwargs):
        self.env = {class_name}(**kwargs)
        # Set render mode if supported
        if hasattr(self.env, 'render_mode'):
            self.env.render_mode = render_mode
    
    def __getattr__(self, name):
        return getattr(self.env, name)

def make_env(env_name=None, render_mode=None, **kwargs):
    return DirectWrapper(render_mode=render_mode or "rgb_array", **kwargs)
'''

    return config_code


def main():
    """Main function"""
    print("Fixed Tetris Gymnasium Diagnostic")
    print("=" * 50)

    solution = create_working_solution()

    if solution:
        config_code = generate_config_code(solution)

        print("\n" + "="*50)
        print("SUCCESS! Working configuration found:")
        print("="*50)
        print(config_code)

        # Save to file
        with open("working_config.py", "w") as f:
            f.write(config_code)
        print(f"\n💾 Saved working config to: working_config.py")
        print("\nNext steps:")
        print("1. Replace your config.py with working_config.py")
        print("2. Run a simple test to verify it works")

    else:
        print("\n❌ No working solution found!")
        print("This tetris-gymnasium installation might be broken.")
        print("Try:")
        print("1. pip uninstall tetris-gymnasium")
        print("2. pip install tetris-gymnasium")
        print("3. Or try a different Tetris environment")


if __name__ == "__main__":
    main()
