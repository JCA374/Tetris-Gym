#!/usr/bin/env python3
"""
Test script to debug Tetris Gymnasium rendering issues
"""

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import time


def test_raw_rendering():
    """Test raw Tetris Gymnasium rendering"""
    print("Testing raw Tetris Gymnasium rendering...")

    # Register environment
    try:
        register(
            id="TetrisRenderTest-v0",
            entry_point="tetris_gymnasium.envs.tetris:Tetris",
        )
    except gym.error.Error:
        pass  # Already registered

    # Test human mode
    print("\n1. Testing human render mode...")
    try:
        env = gym.make("TetrisRenderTest-v0", render_mode="human")
        print(f"✅ Environment created with human mode")

        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Observation type: {type(obs)}")
        if isinstance(obs, dict):
            print(f"   Dict keys: {list(obs.keys())}")
            if 'board' in obs:
                print(f"   Board shape: {obs['board'].shape}")

        # Take a few steps and render
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"   Step {step+1}: Action={action}, Reward={reward:.3f}")

            # Force render
            try:
                env.render()
                print(f"   ✅ Render call successful")
            except Exception as e:
                print(f"   ❌ Render failed: {e}")

            time.sleep(0.5)  # Slow down for visibility

            if terminated or truncated:
                print(f"   Episode ended at step {step+1}")
                break

        env.close()
        print("✅ Human mode test completed\n")

    except Exception as e:
        print(f"❌ Human mode test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test RGB array mode
    print("2. Testing rgb_array render mode...")
    try:
        env = gym.make("TetrisRenderTest-v0", render_mode="rgb_array")
        print(f"✅ Environment created with rgb_array mode")

        obs, info = env.reset()

        # Get RGB array
        rgb_array = env.render()
        if rgb_array is not None:
            print(
                f"✅ RGB array obtained: shape={rgb_array.shape}, dtype={rgb_array.dtype}")
            print(f"   RGB range: [{rgb_array.min()}, {rgb_array.max()}]")

            # Save image for inspection
            try:
                from PIL import Image
                img = Image.fromarray(rgb_array)
                img.save("tetris_render_test.png")
                print(f"✅ Test image saved: tetris_render_test.png")
            except ImportError:
                print("   PIL not available, couldn't save image")
        else:
            print(f"❌ RGB array is None")

        env.close()
        print("✅ RGB array mode test completed\n")

    except Exception as e:
        print(f"❌ RGB array mode test failed: {e}")
        import traceback
        traceback.print_exc()


def test_manual_game():
    """Test manual game with keyboard input"""
    print("3. Testing manual game (if supported)...")

    try:
        register(
            id="TetrisManualTest-v0",
            entry_point="tetris_gymnasium.envs.tetris:Tetris",
        )
    except gym.error.Error:
        pass

    try:
        env = gym.make("TetrisManualTest-v0", render_mode="human")

        print("Manual game test - taking random actions...")
        print("Action meanings (typical Tetris):")
        print("  0: No-op, 1: Right, 2: Left, 3: Down, 4: Rotate CW, 5: Rotate CCW, 6: Hard Drop")

        obs, info = env.reset()

        for step in range(30):  # Take 30 random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {step+1:2d}: Action={action} ({get_action_name(action)}), "
                  f"Reward={reward:6.3f}, Done={terminated or truncated}")

            env.render()
            time.sleep(0.3)  # Slower for better visibility

            if terminated or truncated:
                print("Game ended!")
                obs, info = env.reset()
                print("Game reset for next round...")

        env.close()
        print("✅ Manual game test completed")

    except Exception as e:
        print(f"❌ Manual game test failed: {e}")


def get_action_name(action):
    """Get human-readable action name"""
    action_names = {
        0: "No-op",
        1: "Right",
        2: "Left",
        3: "Down",
        4: "Rotate CW",
        5: "Rotate CCW",
        6: "Hard Drop",
        7: "Hold"
    }
    return action_names.get(action, f"Unknown({action})")


def test_pygame_backend():
    """Test if pygame is available and working"""
    print("4. Testing pygame backend...")

    try:
        import pygame
        print(f"✅ Pygame available: version {pygame.version.ver}")

        # Test pygame initialization
        pygame.init()
        print("✅ Pygame initialized successfully")

        # Test display
        display_info = pygame.display.Info()
        print(f"✅ Display info: {display_info.bitsize}bit, {display_info.fmt}")

        pygame.quit()
        print("✅ Pygame test completed")

    except ImportError:
        print("❌ Pygame not available - this might be the rendering issue!")
        print("   Try: pip install pygame")
    except Exception as e:
        print(f"❌ Pygame test failed: {e}")


def test_environment_backends():
    """Test different rendering backends"""
    print("5. Testing different environment configurations...")

    try:
        register(
            id="TetrisBackendTest-v0",
            entry_point="tetris_gymnasium.envs.tetris:Tetris",
        )
    except gym.error.Error:
        pass

    render_modes = ["human", "rgb_array"]

    for mode in render_modes:
        print(f"\nTesting render_mode='{mode}'...")
        try:
            env = gym.make("TetrisBackendTest-v0", render_mode=mode)
            obs, info = env.reset()

            # Test a few steps
            for i in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if mode == "rgb_array":
                    rgb = env.render()
                    if rgb is not None:
                        print(f"  Step {i+1}: RGB shape {rgb.shape}")
                    else:
                        print(f"  Step {i+1}: RGB is None")
                else:
                    env.render()
                    print(f"  Step {i+1}: Human render called")

            env.close()
            print(f"✅ Mode '{mode}' works")

        except Exception as e:
            print(f"❌ Mode '{mode}' failed: {e}")


def main():
    """Run all rendering tests"""
    print("Tetris Gymnasium Rendering Debug")
    print("=" * 50)

    # Test pygame first
    test_pygame_backend()

    # Test basic rendering
    test_raw_rendering()

    # Test different backends
    test_environment_backends()

    # Test manual game
    test_manual_game()

    print("\n" + "=" * 50)
    print("Rendering tests completed!")
    print("\nIf you're still seeing black screens:")
    print("1. Make sure pygame is installed: pip install pygame")
    print("2. Try running in a different terminal or with X11 forwarding")
    print("3. Check if your system supports GUI applications")
    print("4. Try using --no-render flag for training")


if __name__ == "__main__":
    main()
