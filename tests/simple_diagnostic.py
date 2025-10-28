# simple_diagnostic.py
"""
Simplified diagnostic to identify observation and environment issues.
No PyTorch required.
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
from collections import Counter

# Only import environment
try:
    from config import make_env
    print("✅ Config imported successfully")
except Exception as e:
    print(f"❌ Error importing config: {e}")
    sys.exit(1)


def test_environment_basics():
    """Test basic environment functionality"""
    print("\n" + "="*80)
    print("🔍 ENVIRONMENT BASICS TEST")
    print("="*80)
    
    try:
        env = make_env(render_mode="rgb_array", use_complete_vision=True)
        print("✅ Environment created")
        
        # Reset
        obs, info = env.reset(seed=42)
        print(f"\n📊 Initial Observation:")
        print(f"   Shape: {obs.shape}")
        print(f"   Dtype: {obs.dtype}")
        print(f"   Range: [{obs.min():.4f}, {obs.max():.4f}]")
        print(f"   Non-zero: {np.count_nonzero(obs)}/{obs.size} ({100*np.count_nonzero(obs)/obs.size:.1f}%)")
        
        # Critical check
        if len(obs.shape) == 1:
            print("\n❌ CRITICAL PROBLEM: 1D observation!")
            print("   Agent cannot see spatial structure!")
        elif len(obs.shape) == 2:
            print(f"\n✅ 2D observation - board shape: {obs.shape}")
        elif len(obs.shape) == 3:
            print(f"\n✅ 3D observation - shape: {obs.shape}")
        
        env.close()
        return True
    except Exception as e:
        print(f"\n❌ Environment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_dynamics():
    """Test if observations change with actions"""
    print("\n" + "="*80)
    print("🔍 OBSERVATION DYNAMICS TEST")
    print("="*80)
    
    try:
        env = make_env(render_mode="rgb_array", use_complete_vision=True)
        
        obs, _ = env.reset(seed=42)
        initial_obs = obs.copy()
        
        # Take actions and check for changes
        changes = []
        for i in range(10):
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            change = np.abs(next_obs - obs).sum()
            changes.append(change)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        avg_change = np.mean(changes)
        print(f"\n📊 Observation changes:")
        print(f"   Average change per step: {avg_change:.6f}")
        print(f"   Min change: {min(changes):.6f}")
        print(f"   Max change: {max(changes):.6f}")
        
        if avg_change < 0.0001:
            print("\n❌ CRITICAL: Observations are frozen!")
            print("   Agent cannot learn if observations don't change!")
            env.close()
            return False
        
        print("\n✅ Observations change dynamically")
        env.close()
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_random_baseline():
    """Test if random actions can clear any lines"""
    print("\n" + "="*80)
    print("🔍 RANDOM BASELINE TEST")
    print("="*80)
    
    try:
        env = make_env(render_mode="rgb_array", use_complete_vision=True)
        
        total_lines = 0
        total_steps = 0
        total_episodes = 20
        
        print(f"\nRunning {total_episodes} episodes with random actions...")
        
        episodes_with_lines = 0
        
        for episode in range(total_episodes):
            obs, _ = env.reset()
            done = False
            episode_steps = 0
            episode_lines = 0
            
            while not done and episode_steps < 200:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_steps += 1
                
                lines_cleared = info.get('number_of_lines', 0)
                episode_lines += lines_cleared
            
            total_lines += episode_lines
            total_steps += episode_steps
            
            if episode_lines > 0:
                episodes_with_lines += 1
                print(f"  Episode {episode:2d}: {episode_lines} lines, {episode_steps} steps")
        
        avg_lines = total_lines / total_episodes
        avg_steps = total_steps / total_episodes
        
        print(f"\n📊 Random Baseline Results:")
        print(f"   Total lines cleared: {total_lines}")
        print(f"   Episodes with lines: {episodes_with_lines}/{total_episodes}")
        print(f"   Average lines/episode: {avg_lines:.3f}")
        print(f"   Average steps/episode: {avg_steps:.1f}")
        
        if total_lines == 0:
            print("\n❌ CRITICAL: Random policy cleared 0 lines!")
            print("   Environment might be broken or impossible")
            env.close()
            return False
        
        if avg_steps < 15:
            print("\n⚠️  WARNING: Episodes are very short!")
            print("   Agent dies too quickly to learn")
        
        print(f"\n✅ Random policy CAN clear lines")
        print(f"   Your trained agent should do MUCH better than {avg_lines:.3f} lines/ep")
        
        env.close()
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_space():
    """Test action space"""
    print("\n" + "="*80)
    print("🔍 ACTION SPACE TEST")
    print("="*80)
    
    try:
        env = make_env(render_mode="rgb_array", use_complete_vision=True)
        
        print(f"\n📊 Action Space:")
        print(f"   Type: {type(env.action_space)}")
        print(f"   Number of actions: {env.action_space.n}")
        
        # Tetris typically has 8 actions
        action_names = ['NOOP', 'LEFT', 'RIGHT', 'DOWN', 'ROTATE_CW', 'ROTATE_CCW', 'DROP', 'HOLD']
        
        print(f"\n   Actions:")
        for i in range(env.action_space.n):
            name = action_names[i] if i < len(action_names) else f"ACTION_{i}"
            print(f"     {i}: {name}")
        
        env.close()
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    """Run all simple diagnostics"""
    print("\n" + "="*80)
    print("🔬 SIMPLE DIAGNOSTIC FOR ZERO LINES PROBLEM")
    print("="*80)
    print("\nYour agent trained for 5000 episodes but cleared 0 lines.")
    print("Let's run some basic tests...\n")
    
    results = {}
    
    tests = [
        ("Environment Basics", test_environment_basics),
        ("Observation Dynamics", test_observation_dynamics),
        ("Action Space", test_action_space),
        ("Random Baseline", test_random_baseline),
    ]
    
    for name, test_func in tests:
        result = test_func()
        results[name] = result
    
    # Summary
    print("\n" + "="*80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*80)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    
    # Recommendations
    print("\n" + "="*80)
    print("💡 LIKELY PROBLEMS & FIXES")
    print("="*80)
    
    if not results.get("Random Baseline", False):
        print("\n🔴 CRITICAL: Environment cannot clear lines!")
        print("   Possible causes:")
        print("   1. tetris-gymnasium not properly installed")
        print("   2. Wrong environment configuration")
        print("   3. Actions not working correctly")
        print("\n   FIX: Reinstall tetris-gymnasium:")
        print("   pip install --upgrade tetris-gymnasium")
    
    elif not results.get("Observation Dynamics", False):
        print("\n🔴 CRITICAL: Frozen observations!")
        print("   Agent cannot learn if state doesn't change")
        print("\n   FIX: Check observation wrapper in config.py")
    
    else:
        print("\n💡 Environment looks OK. The problem is likely:")
        print("\n1. ⚠️  EPSILON DECAY TOO FAST")
        print("   Your agent might have stopped exploring before learning")
        print("   Current: epsilon_decay = 0.9995")
        print("   After 5000 episodes: epsilon ≈ 0.05")
        print("   FIX: Use slower decay: 0.9998 or 0.9999")
        print("\n2. ⚠️  EPISODES TOO SHORT (12 steps avg)")
        print("   Agent dies before seeing consequences of actions")
        print("   FIX: Increase survival reward")
        print("   FIX: Start with more exploration (epsilon_start = 1.0)")
        print("\n3. ⚠️  REWARD SHAPING NOT STRONG ENOUGH")
        print("   Line bonuses might not be reaching the agent")
        print("   FIX: Increase line bonuses to 1000+ per line")
        print("   FIX: Add small positive reward for each step survived")
    
    print("\n" + "="*80)
    print("📝 NEXT STEPS")
    print("="*80)
    print("""
1. If any tests failed above, fix those issues first
2. Try training with these better hyperparameters:
   - epsilon_decay = 0.9998 (slower exploration decay)
   - epsilon_start = 1.0 (full exploration initially)
   - Survival bonus: +0.1 per step
   - Line bonus: +1000 per line minimum

3. Run: python train.py --episodes 2000

4. Monitor early performance (first 500 episodes):
   - You should see SOME lines cleared
   - Episode length should increase over time
   - If still 0 lines after 500 episodes, stop and debug

5. Expected trajectory:
   - Episode 100-500: First line cleared
   - Episode 1000: 0.5-2.0 lines/episode
   - Episode 5000: 5-15 lines/episode
    """)


if __name__ == "__main__":
    main()
