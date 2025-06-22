#!/usr/bin/env python3
"""
test_piece_visibility.py

Quick test to verify your model can see the active piece
This will show you exactly what information was missing!
"""

import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
import gymnasium as gym

def test_piece_visibility():
    """Test what piece information is available and being used"""
    print("🔍 TESTING TETRIS PIECE VISIBILITY")
    print("="*60)
    
    # Register environment
    try:
        register(
            id="TetrisPieceTest-v0",
            entry_point="tetris_gymnasium.envs.tetris:Tetris",
        )
    except gym.error.Error:
        pass
    
    # Create raw environment
    env = gym.make("TetrisPieceTest-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    print("📊 AVAILABLE INFORMATION:")
    print(f"   Observation keys: {list(obs.keys())}")
    
    for key, value in obs.items():
        print(f"   {key}: {value.shape} {value.dtype}")
        print(f"      Range: [{value.min()}, {value.max()}]")
        print(f"      Non-zero elements: {np.sum(value != 0)}")
        print()
    
    # Test what happens during piece placement
    print("🎮 TESTING PIECE MOVEMENT:")
    active_piece_changes = []
    board_changes = []
    
    initial_active = obs['active_tetromino_mask'].copy()
    initial_board = obs['board'].copy()
    
    for step in range(10):
        # Try different actions
        action = step % 8  # Cycle through all actions
        new_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check for changes
        active_change = np.sum(np.abs(new_obs['active_tetromino_mask'] - initial_active))
        board_change = np.sum(np.abs(new_obs['board'] - initial_board))
        
        active_piece_changes.append(active_change)
        board_changes.append(board_change)
        
        print(f"   Step {step+1}: Action={action}, Active change={active_change:.1f}, Board change={board_change:.1f}")
        
        # Update for next comparison
        initial_active = new_obs['active_tetromino_mask'].copy()
        initial_board = new_obs['board'].copy()
        
        if terminated or truncated:
            new_obs, info = env.reset()
            initial_active = new_obs['active_tetromino_mask'].copy()
            initial_board = new_obs['board'].copy()
            print("   (Game reset)")
    
    avg_active_change = np.mean(active_piece_changes)
    avg_board_change = np.mean(board_changes)
    
    print(f"\n📊 PIECE DYNAMICS:")
    print(f"   Average active piece change: {avg_active_change:.2f}")
    print(f"   Average board change: {avg_board_change:.2f}")
    
    if avg_active_change > 1.0:
        print("   ✅ Active piece information is DYNAMIC and meaningful!")
    else:
        print("   ❌ Active piece information barely changes")
    
    # Test your current vs complete observation
    print(f"\n🔬 OBSERVATION COMPARISON:")
    
    # Reset for consistent comparison
    obs, info = env.reset(seed=42)
    
    # Your current approach (board only)
    current_obs = obs['board'].astype(np.float32)
    if current_obs.max() > 1:
        current_obs = current_obs / current_obs.max()
    current_info = current_obs.size
    
    print(f"   Current (board-only): {current_obs.shape} = {current_info} features")
    print(f"      Range: [{current_obs.min():.3f}, {current_obs.max():.3f}]")
    
    # Complete approach (board + pieces)
    board_norm = obs['board'].astype(np.float32)
    if board_norm.max() > 1:
        board_norm = board_norm / board_norm.max()
    
    active_norm = obs['active_tetromino_mask'].astype(np.float32)
    holder_norm = obs['holder'].astype(np.float32)
    queue_norm = obs['queue'].astype(np.float32)
    
    # Try to stack them (simplified version)
    try:
        if active_norm.shape == board_norm.shape:
            complete_obs = np.stack([board_norm, active_norm], axis=-1)
            print(f"   Complete (board+active): {complete_obs.shape} = {complete_obs.size} features")
            print(f"      Board channel range: [{complete_obs[:,:,0].min():.3f}, {complete_obs[:,:,0].max():.3f}]")
            print(f"      Active channel range: [{complete_obs[:,:,1].min():.3f}, {complete_obs[:,:,1].max():.3f}]")
            
            # Check if active piece adds information
            active_pixels = np.sum(complete_obs[:,:,1] > 0.01)
            print(f"      Active piece pixels: {active_pixels}")
            
            if active_pixels > 0:
                print("   ✅ Active piece adds valuable information!")
                
                # Show where the active piece is
                board_pixels = np.sum(complete_obs[:,:,0] > 0.01)
                print(f"      Board pixels: {board_pixels}")
                print(f"      Information gain: +{active_pixels} pixels ({active_pixels/board_pixels*100:.1f}% more)")
            else:
                print("   ❌ Active piece channel is empty")
        else:
            print(f"   ⚠️  Shape mismatch: board {board_norm.shape} vs active {active_norm.shape}")
            
    except Exception as e:
        print(f"   ❌ Error combining observations: {e}")
    
    # Visual comparison
    create_visual_comparison(obs)
    
    env.close()
    
    return avg_active_change > 1.0


def create_visual_comparison(obs):
    """Create visual comparison of board vs complete observation"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Tetris Observation Comparison: What Your Model Was Missing!', fontsize=16)
        
        # Board only (what your model currently sees)
        board = obs['board']
        axes[0,0].imshow(board, cmap='viridis', aspect='auto')
        axes[0,0].set_title('Current: Board Only\n(What your model sees)')
        axes[0,0].set_xlabel('Width')
        axes[0,0].set_ylabel('Height')
        
        # Active piece (what was missing!)
        active = obs['active_tetromino_mask']
        axes[0,1].imshow(active, cmap='Reds', aspect='auto')
        axes[0,1].set_title('MISSING: Active Piece\n(Critical for placement!)')
        active_pixels = np.sum(active > 0)
        axes[0,1].set_xlabel(f'{active_pixels} active pixels')
        
        # Holder
        holder = obs['holder']
        axes[0,2].imshow(holder, cmap='Blues', aspect='auto')
        axes[0,2].set_title('MISSING: Held Piece\n(Strategic planning)')
        
        # Combined view
        if active.shape == board.shape:
            combined = np.zeros((*board.shape, 3))
            combined[:,:,0] = board / max(1, board.max())  # Red = board
            combined[:,:,1] = active / max(1, active.max())  # Green = active
            axes[1,0].imshow(combined)
            axes[1,0].set_title('Combined: Board + Active Piece\n(What model SHOULD see)')
        else:
            axes[1,0].text(0.5, 0.5, f'Shape mismatch:\nBoard: {board.shape}\nActive: {active.shape}', 
                          transform=axes[1,0].transAxes, ha='center', va='center')
            axes[1,0].set_title('Shape Mismatch Issue')
        
        # Queue preview
        queue = obs['queue']
        if queue.ndim >= 2:
            if queue.ndim == 3:
                next_piece = queue[0]
            else:
                next_piece = queue
            axes[1,1].imshow(next_piece, cmap='Oranges', aspect='auto')
            axes[1,1].set_title('MISSING: Next Pieces\n(Forward planning)')
        
        # Information density comparison
        info_current = board.size
        info_complete = board.size + active.size + holder.size + queue.size
        
        categories = ['Current\n(Board Only)', 'Complete\n(All Info)']
        info_amounts = [info_current, info_complete]
        
        axes[1,2].bar(categories, info_amounts, color=['red', 'green'])
        axes[1,2].set_title('Information Available')
        axes[1,2].set_ylabel('Features')
        
        # Add text annotations
        for i, v in enumerate(info_amounts):
            axes[1,2].text(i, v + max(info_amounts)*0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('piece_visibility_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Visual analysis saved as 'piece_visibility_analysis.png'")
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


def main():
    """Main test function"""
    print("Testing if your model can see the pieces it's supposed to place...")
    print("This might explain your 62k episode plateau!")
    print()
    
    piece_info_available = test_piece_visibility()
    
    print("\n" + "="*60)
    print("🎯 PIECE VISIBILITY TEST RESULTS")
    print("="*60)
    
    if piece_info_available:
        print("✅ PIECE INFORMATION IS AVAILABLE!")
        print("\nYour model has been missing critical information:")
        print("• Current piece shape and rotation")  
        print("• Piece position on board")
        print("• Strategic planning information")
        print("\n🔧 SOLUTION:")
        print("1. Replace config.py with config_with_piece_vision.py")
        print("2. Retrain with complete observation")
        print("3. Expect DRAMATIC improvement in line clearing!")
        
    else:
        print("❌ PIECE INFORMATION ISSUE")
        print("The active piece information may not be working properly.")
        print("This could indicate an environment or observation issue.")
    
    print(f"\n📊 Check 'piece_visibility_analysis.png' for visual proof!")
    
    return piece_info_available


if __name__ == "__main__":
    main()