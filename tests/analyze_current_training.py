#!/usr/bin/env python3
"""
analyze_current_training.py

Analyze current training state and logs to identify why training is stuck
"""

import os
import json
import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

def find_training_files():
    """Find all training-related files and checkpoints"""
    files_found = {}
    
    # Look for models
    model_dirs = ['models/', 'checkpoints/']
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            checkpoints = glob.glob(os.path.join(model_dir, '*.pth'))
            pickle_files = glob.glob(os.path.join(model_dir, '*.pkl'))
            files_found[f'{model_dir}_checkpoints'] = checkpoints
            files_found[f'{model_dir}_pickles'] = pickle_files
    
    # Look for logs
    log_dirs = ['logs/', 'diagnostics_output/', '.']
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            csv_files = glob.glob(os.path.join(log_dir, '*.csv'))
            json_files = glob.glob(os.path.join(log_dir, '*.json'))
            log_files = glob.glob(os.path.join(log_dir, '*.log'))
            files_found[f'{log_dir}_csv'] = csv_files
            files_found[f'{log_dir}_json'] = json_files  
            files_found[f'{log_dir}_logs'] = log_files
    
    return files_found

def analyze_checkpoint():
    """Analyze the latest checkpoint to understand training state"""
    print("🔍 ANALYZING CHECKPOINT STATE")
    print("="*50)
    
    checkpoint_paths = [
        'models/latest_checkpoint.pth',
        'models/latest_model.pth', 
        'checkpoints/latest_checkpoint.pth'
    ]
    
    checkpoint_found = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_found = path
            break
    
    if not checkpoint_found:
        print("❌ No checkpoint found")
        print("   Paths checked:", checkpoint_paths)
        return None
    
    print(f"✅ Found checkpoint: {checkpoint_found}")
    
    try:
        import torch
        checkpoint = torch.load(checkpoint_found, map_location='cpu', weights_only=False)
        
        print(f"\n📊 CHECKPOINT ANALYSIS:")
        print(f"   Episodes completed: {checkpoint.get('episode', 'Unknown')}")
        print(f"   Steps completed: {checkpoint.get('steps_done', 'Unknown')}")
        print(f"   Current epsilon: {checkpoint.get('epsilon', 'Unknown'):.6f}")
        
        # Analyze recent rewards
        total_rewards = checkpoint.get('total_rewards', [])
        if total_rewards:
            recent_rewards = total_rewards[-100:] if len(total_rewards) >= 100 else total_rewards
            print(f"   Total episodes trained: {len(total_rewards)}")
            print(f"   Average reward (last 100): {np.mean(recent_rewards):.2f}")
            print(f"   Max reward achieved: {np.max(total_rewards):.2f}")
            print(f"   Min reward: {np.min(total_rewards):.2f}")
            print(f"   Recent reward std: {np.std(recent_rewards):.2f}")
        
        # Analyze episode metrics if available
        episode_metrics = checkpoint.get('episode_metrics', [])
        if episode_metrics:
            recent_metrics = episode_metrics[-100:] if len(episode_metrics) >= 100 else episode_metrics
            lines_cleared = [m.get('lines_cleared', 0) for m in recent_metrics if isinstance(m, dict)]
            if lines_cleared:
                total_lines = sum(lines_cleared)
                print(f"   Total lines cleared (last 100 eps): {total_lines}")
                print(f"   Average lines per episode: {total_lines/len(lines_cleared):.3f}")
                episodes_with_lines = sum(1 for l in lines_cleared if l > 0)
                print(f"   Episodes with lines cleared: {episodes_with_lines}/{len(lines_cleared)}")
        
        # Check reward shaping
        reward_shaping = checkpoint.get('reward_shaping_type', 'Unknown')
        print(f"   Reward shaping: {reward_shaping}")
        
        # Check epsilon decay method
        epsilon_method = checkpoint.get('epsilon_decay_method', 'Unknown')
        max_episodes = checkpoint.get('max_episodes', 'Unknown')
        print(f"   Epsilon method: {epsilon_method}")
        print(f"   Max episodes: {max_episodes}")
        
        return {
            'episode': checkpoint.get('episode', 0),
            'epsilon': checkpoint.get('epsilon', 0),
            'total_rewards': total_rewards,
            'episode_metrics': episode_metrics,
            'reward_shaping': reward_shaping,
            'epsilon_method': epsilon_method
        }
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def analyze_training_logs():
    """Analyze training logs to identify patterns and issues"""
    print("\n🔍 ANALYZING TRAINING LOGS")
    print("="*50)
    
    # Look for CSV files with episode data
    csv_files = []
    for pattern in ['*episode*.csv', '*training*.csv', '*tetris*.csv', '*.csv']:
        csv_files.extend(glob.glob(pattern))
        csv_files.extend(glob.glob(f'logs/{pattern}'))
        csv_files.extend(glob.glob(f'diagnostics_output/{pattern}'))
    
    csv_files = list(set(csv_files))  # Remove duplicates
    
    if not csv_files:
        print("❌ No training CSV files found")
        return None
    
    print(f"📊 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   • {f}")
    
    # Analyze the most recent/largest CSV file
    largest_csv = max(csv_files, key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0)
    print(f"\n📈 Analyzing: {largest_csv}")
    
    try:
        import pandas as pd
        df = pd.read_csv(largest_csv)
        
        print(f"\n📊 TRAINING LOG ANALYSIS:")
        print(f"   Total episodes logged: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Analyze rewards
        if 'reward' in df.columns:
            recent_rewards = df['reward'].tail(100)
            print(f"   Recent avg reward: {recent_rewards.mean():.2f}")
            print(f"   Recent reward trend: {recent_rewards.diff().mean():.4f}/episode")
            
        # Analyze lines cleared
        lines_cols = [col for col in df.columns if 'lines' in col.lower()]
        if lines_cols:
            lines_col = lines_cols[0]
            recent_lines = df[lines_col].tail(100)
            total_lines = recent_lines.sum()
            print(f"   Lines cleared (last 100): {total_lines}")
            print(f"   Avg lines per episode: {total_lines/len(recent_lines):.3f}")
            
        # Analyze epsilon
        if 'epsilon' in df.columns:
            recent_epsilon = df['epsilon'].tail(10)
            print(f"   Current epsilon: {recent_epsilon.iloc[-1]:.6f}")
            print(f"   Epsilon trend: {recent_epsilon.diff().mean():.8f}/episode")
            
        # Check for plateau (no improvement)
        if 'reward' in df.columns and len(df) >= 200:
            last_200 = df['reward'].tail(200)
            first_half = last_200[:100].mean()
            second_half = last_200[100:].mean()
            improvement = second_half - first_half
            print(f"   Improvement (last 200 eps): {improvement:.2f}")
            
            if abs(improvement) < 1.0:
                print("   🚨 PLATEAU DETECTED: No significant improvement!")
            elif improvement < 0:
                print("   🚨 REGRESSION DETECTED: Performance getting worse!")
        
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        return None

def create_training_plots(df, checkpoint_data):
    """Create diagnostic plots from training data"""
    if df is None:
        return
        
    print("\n📊 CREATING DIAGNOSTIC PLOTS")
    print("="*50)
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tetris Training Diagnostic Analysis', fontsize=16)
        
        # Plot 1: Reward over time
        if 'reward' in df.columns:
            axes[0,0].plot(df.index, df['reward'], alpha=0.6, label='Episode Reward')
            # Add moving average
            window = min(50, len(df)//10)
            if window > 1:
                moving_avg = df['reward'].rolling(window=window).mean()
                axes[0,0].plot(df.index, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            axes[0,0].set_title('Reward Progress')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Reward')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Lines cleared
        lines_cols = [col for col in df.columns if 'lines' in col.lower()]
        if lines_cols:
            lines_col = lines_cols[0]
            axes[0,1].plot(df.index, df[lines_col], 'o-', alpha=0.7, markersize=2)
            axes[0,1].set_title('Lines Cleared per Episode')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Lines Cleared')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Epsilon decay
        if 'epsilon' in df.columns:
            axes[0,2].plot(df.index, df['epsilon'])
            axes[0,2].set_title('Epsilon Decay')
            axes[0,2].set_xlabel('Episode')
            axes[0,2].set_ylabel('Epsilon')
            axes[0,2].set_yscale('log')
            axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Reward distribution
        if 'reward' in df.columns:
            axes[1,0].hist(df['reward'], bins=50, alpha=0.7, edgecolor='black')
            axes[1,0].axvline(df['reward'].mean(), color='red', linestyle='--', 
                             label=f'Mean: {df["reward"].mean():.1f}')
            axes[1,0].set_title('Reward Distribution')
            axes[1,0].set_xlabel('Reward')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Episode length if available
        if 'steps' in df.columns:
            axes[1,1].plot(df.index, df['steps'], alpha=0.6)
            window = min(50, len(df)//10)
            if window > 1:
                moving_avg = df['steps'].rolling(window=window).mean()
                axes[1,1].plot(df.index, moving_avg, 'r-', linewidth=2)
            axes[1,1].set_title('Episode Length')
            axes[1,1].set_xlabel('Episode')
            axes[1,1].set_ylabel('Steps')
            axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Learning progress (reward trend)
        if 'reward' in df.columns and len(df) >= 100:
            window = 100
            trend = df['reward'].rolling(window=window).mean()
            axes[1,2].plot(df.index[window-1:], trend[window-1:])
            axes[1,2].set_title(f'Learning Trend ({window}-episode average)')
            axes[1,2].set_xlabel('Episode')
            axes[1,2].set_ylabel('Average Reward')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_diagnostic_plots.png', dpi=150, bbox_inches='tight')
        print("✅ Plots saved as 'training_diagnostic_plots.png'")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")

def diagnose_stuck_training(checkpoint_data, df):
    """Diagnose why training might be stuck"""
    print("\n🔧 DIAGNOSING STUCK TRAINING")
    print("="*50)
    
    issues = []
    recommendations = []
    
    if checkpoint_data:
        # Check epsilon
        epsilon = checkpoint_data.get('epsilon', 1.0)
        episode = checkpoint_data.get('episode', 0)
        
        if epsilon < 0.01 and episode < 5000:
            issues.append("🚨 EPSILON TOO LOW: Epsilon collapsed too early")
            recommendations.append("   Fix: Increase epsilon_end to 0.05-0.1")
            recommendations.append("   Fix: Slower epsilon decay (0.9995 instead of 0.995)")
        
        if epsilon > 0.5 and episode > 1000:
            issues.append("⚠️  EPSILON TOO HIGH: Not exploiting learned knowledge")
            recommendations.append("   Fix: Faster epsilon decay or lower starting epsilon")
        
        # Check episode count vs max episodes
        max_episodes = checkpoint_data.get('max_episodes', 10000)
        if episode >= max_episodes * 0.9:
            issues.append("⚠️  NEAR EPISODE LIMIT: Training may stop soon")
            recommendations.append("   Fix: Increase max_episodes in training")
    
    if df is not None:
        # Check for plateau
        if 'reward' in df.columns and len(df) >= 200:
            last_200 = df['reward'].tail(200)
            improvement = last_200.tail(100).mean() - last_200.head(100).mean()
            
            if abs(improvement) < 0.5:
                issues.append("🚨 REWARD PLATEAU: No improvement in last 200 episodes")
                recommendations.append("   Fix: Run plateau breaker: python break_plateau_train.py")
                recommendations.append("   Fix: Increase reward shaping weights")
                recommendations.append("   Fix: Boost epsilon temporarily")
        
        # Check lines cleared
        lines_cols = [col for col in df.columns if 'lines' in col.lower()]
        if lines_cols:
            lines_col = lines_cols[0]
            recent_lines = df[lines_col].tail(100).sum()
            if recent_lines == 0:
                issues.append("🚨 ZERO LINES CLEARED: Agent not learning core objective")
                recommendations.append("   CRITICAL: Run vision diagnostic to check if agent can see board")
                recommendations.append("   Fix: python visual_board_check.py")
                recommendations.append("   Fix: Increase line clear bonuses to 50-100")
            elif recent_lines < 10:
                issues.append("⚠️  FEW LINES CLEARED: Learning but slowly")
                recommendations.append("   Fix: Stronger reward shaping for line clears")
                recommendations.append("   Fix: Action masking to avoid useless actions")
    
    # Print diagnosis
    if issues:
        print("📋 ISSUES IDENTIFIED:")
        for issue in issues:
            print(f"   {issue}")
        print("\n🔧 RECOMMENDED FIXES:")
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("✅ No obvious issues detected")
        print("   Training parameters appear reasonable")
        print("   Issue may be in model architecture or environment")
    
    return issues, recommendations

def main():
    """Main analysis function"""
    print("🔍 TETRIS TRAINING ANALYSIS")
    print("="*80)
    print("Analyzing current training state to identify why you're stuck...")
    print()
    
    # Find all training files
    files_found = find_training_files()
    print("📁 TRAINING FILES FOUND:")
    for category, files in files_found.items():
        if files:
            print(f"   {category}: {len(files)} files")
            for f in files[:3]:  # Show first 3
                print(f"     • {f}")
            if len(files) > 3:
                print(f"     ... and {len(files)-3} more")
    print()
    
    # Analyze checkpoint
    checkpoint_data = analyze_checkpoint()
    
    # Analyze training logs  
    df = analyze_training_logs()
    
    # Create diagnostic plots
    create_training_plots(df, checkpoint_data)
    
    # Diagnose issues
    issues, recommendations = diagnose_stuck_training(checkpoint_data, df)
    
    print("\n" + "="*80)
    print("🎯 ANALYSIS COMPLETE")
    print("="*80)
    
    if issues:
        print("❌ TRAINING IS STUCK - Issues identified above")
        print("\n📋 IMMEDIATE ACTIONS:")
        print("   1. Run vision diagnostic: python run_complete_diagnostics.py")
        print("   2. Check if model can see board: python visual_board_check.py") 
        print("   3. If vision OK, run plateau breaker: python break_plateau_train.py")
        print("   4. Check training_diagnostic_plots.png for visual analysis")
    else:
        print("✅ TRAINING LOOKS HEALTHY")
        print("   If still concerned, run: python run_complete_diagnostics.py")
    
    return len(issues) == 0

if __name__ == "__main__":
    main()