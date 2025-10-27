# diagnose_training.py
"""
Comprehensive diagnostic tool to analyze Tetris RL training status
and determine whether to continue training or fix code issues.
"""

import os
import sys
import glob
import json
import pickle
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - checkpoint analysis limited")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Pandas not available - CSV analysis limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TrainingDiagnostic:
    def __init__(self):
        self.checkpoint_data = None
        self.log_data = None
        self.issues = []
        self.recommendations = []
        
    def find_files(self):
        """Find all training-related files"""
        print("\n🔍 SEARCHING FOR TRAINING FILES...")
        print("=" * 60)
        
        files_found = {
            'checkpoints': [],
            'logs': [],
            'configs': []
        }
        
        # Search for checkpoints
        for pattern in ['models/*.pth', 'models/*.pkl', 'checkpoints/*.pth', '*.pth']:
            files_found['checkpoints'].extend(glob.glob(pattern))
        
        # Search for logs
        for pattern in ['logs/**/*.csv', 'logs/**/*.json', '*.csv', '*.json']:
            files_found['logs'].extend(glob.glob(pattern, recursive=True))
        
        # Search for configs
        for pattern in ['config.json', 'config.py', '*/config.json']:
            files_found['configs'].extend(glob.glob(pattern))
        
        # Print findings
        for category, file_list in files_found.items():
            if file_list:
                print(f"\n✅ Found {len(file_list)} {category}:")
                for f in sorted(set(file_list))[:10]:  # Show first 10
                    size = os.path.getsize(f) / 1024 if os.path.exists(f) else 0
                    print(f"   • {f} ({size:.1f} KB)")
            else:
                print(f"\n❌ No {category} found")
        
        return files_found
    
    def analyze_checkpoint(self, checkpoint_path):
        """Analyze a training checkpoint"""
        if not TORCH_AVAILABLE:
            print("\n⚠️  Cannot analyze checkpoint without PyTorch")
            return None
        
        print(f"\n📊 ANALYZING CHECKPOINT: {checkpoint_path}")
        print("=" * 60)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract key metrics
            episode = checkpoint.get('episode', checkpoint.get('episodes_done', 0))
            epsilon = checkpoint.get('epsilon', 'Unknown')
            steps = checkpoint.get('steps_done', 'Unknown')
            
            print(f"\n📈 Training Progress:")
            print(f"   Episode: {episode}")
            print(f"   Total Steps: {steps}")
            print(f"   Current Epsilon: {epsilon if isinstance(epsilon, str) else f'{epsilon:.6f}'}")
            
            # Check hyperparameters
            if 'hyperparameters' in checkpoint or 'config' in checkpoint:
                hyper = checkpoint.get('hyperparameters', checkpoint.get('config', {}))
                print(f"\n⚙️  Hyperparameters:")
                for key in ['lr', 'gamma', 'batch_size', 'epsilon_decay']:
                    if key in hyper:
                        print(f"   {key}: {hyper[key]}")
            
            # Check reward info
            if 'total_rewards' in checkpoint:
                rewards = checkpoint['total_rewards']
                if len(rewards) > 0:
                    recent = rewards[-100:] if len(rewards) >= 100 else rewards
                    print(f"\n💰 Reward Statistics:")
                    print(f"   Total episodes logged: {len(rewards)}")
                    print(f"   Recent average (last 100): {sum(recent)/len(recent):.2f}")
                    print(f"   Best reward: {max(rewards):.2f}")
                    print(f"   Worst reward: {min(rewards):.2f}")
            
            # Check for episode metrics
            if 'episode_metrics' in checkpoint:
                metrics = checkpoint['episode_metrics']
                if metrics:
                    recent_metrics = metrics[-100:] if len(metrics) >= 100 else metrics
                    lines_cleared = [m.get('lines_cleared', 0) for m in recent_metrics if isinstance(m, dict)]
                    if lines_cleared:
                        total_lines = sum(lines_cleared)
                        print(f"\n🎯 Line Clearing Performance:")
                        print(f"   Total lines (last 100 eps): {total_lines}")
                        print(f"   Average per episode: {total_lines/len(lines_cleared):.3f}")
                        print(f"   Episodes with lines: {sum(1 for l in lines_cleared if l > 0)}/{len(lines_cleared)}")
            
            self.checkpoint_data = checkpoint
            return checkpoint
            
        except Exception as e:
            print(f"\n❌ Error loading checkpoint: {e}")
            return None
    
    def analyze_logs(self, log_path):
        """Analyze training log CSV"""
        if not PANDAS_AVAILABLE:
            print("\n⚠️  Cannot analyze logs without pandas")
            return None
        
        print(f"\n📊 ANALYZING LOG FILE: {log_path}")
        print("=" * 60)
        
        try:
            df = pd.read_csv(log_path)
            
            print(f"\n📈 Training Statistics:")
            print(f"   Total episodes logged: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Analyze rewards
            if 'reward' in df.columns:
                recent = df['reward'].tail(100)
                print(f"\n💰 Reward Analysis:")
                print(f"   Recent average: {recent.mean():.2f}")
                print(f"   Recent std dev: {recent.std():.2f}")
                print(f"   Reward trend: {recent.diff().mean():.4f} per episode")
                
                # Check for improvement
                if len(df) >= 200:
                    first_half = df['reward'].head(100).mean()
                    second_half = df['reward'].tail(100).mean()
                    improvement = second_half - first_half
                    print(f"   Improvement (100→100): {improvement:.2f}")
                    
                    if improvement < 0:
                        self.issues.append("🚨 REGRESSION: Performance getting worse!")
                    elif abs(improvement) < 1.0:
                        self.issues.append("⚠️  PLATEAU: No significant improvement")
            
            # Analyze lines cleared
            lines_cols = [col for col in df.columns if 'lines' in col.lower()]
            if lines_cols:
                lines_col = lines_cols[0]
                recent_lines = df[lines_col].tail(100)
                total_lines = recent_lines.sum()
                print(f"\n🎯 Line Clearing Analysis:")
                print(f"   Lines cleared (last 100): {total_lines}")
                print(f"   Average per episode: {total_lines/100:.3f}")
                
                if total_lines == 0 and len(df) > 500:
                    self.issues.append("🚨 ZERO LINES: Agent not learning objective!")
            
            # Analyze epsilon
            if 'epsilon' in df.columns:
                recent_epsilon = df['epsilon'].tail(10)
                current_epsilon = recent_epsilon.iloc[-1]
                episode_num = len(df)
                print(f"\n🎲 Exploration Analysis:")
                print(f"   Current epsilon: {current_epsilon:.6f}")
                print(f"   At episode: {episode_num}")
                
                if current_epsilon < 0.05 and episode_num < 2000:
                    self.issues.append("🚨 EPSILON COLLAPSED: Dropped too fast!")
                    self.recommendations.append("   Fix: Increase epsilon_end to 0.05-0.10")
                    self.recommendations.append("   Fix: Use slower decay (0.9999)")
            
            # Analyze episode length
            if 'steps' in df.columns:
                recent_steps = df['steps'].tail(100)
                print(f"\n⏱️  Episode Length Analysis:")
                print(f"   Recent average: {recent_steps.mean():.1f} steps")
                
                if recent_steps.mean() < 20 and len(df) > 500:
                    self.issues.append("⚠️  SHORT EPISODES: Dying too quickly")
            
            self.log_data = df
            return df
            
        except Exception as e:
            print(f"\n❌ Error loading log: {e}")
            return None
    
    def diagnose_issues(self):
        """Diagnose training issues and provide recommendations"""
        print("\n" + "=" * 60)
        print("🔧 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        if not self.checkpoint_data and not self.log_data:
            print("\n❌ No data available for diagnosis")
            print("\nPlease ensure you have either:")
            print("  • A checkpoint file (models/*.pth)")
            print("  • A training log (logs/**/*.csv)")
            return
        
        # Display all issues
        if self.issues:
            print("\n🚨 ISSUES DETECTED:")
            for issue in self.issues:
                print(f"   {issue}")
        else:
            print("\n✅ No major issues detected!")
        
        # Display recommendations
        if self.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in self.recommendations:
                print(f"   {rec}")
        
        # Overall decision
        print("\n" + "=" * 60)
        print("📋 DECISION:")
        print("=" * 60)
        
        critical_issues = [i for i in self.issues if '🚨' in i]
        
        if critical_issues:
            print("\n🛑 STOP TRAINING - FIX CODE FIRST")
            print("\nCritical issues that need fixing:")
            for issue in critical_issues:
                print(f"   • {issue}")
            
            print("\n📝 Next steps:")
            print("   1. Fix the issues listed above")
            print("   2. Consider starting fresh training")
            print("   3. Or use: python break_plateau_training.py")
        else:
            print("\n✅ CONTINUE TRAINING")
            print("\nYour training appears to be progressing normally.")
            print("Small progress is expected in early episodes (< 2000).")
            print("\nContinue training and monitor for:")
            print("   • Gradual reward increase")
            print("   • Occasional line clears starting around ep 1000-2000")
            print("   • Epsilon decay to 0.05-0.10 by episode 5000")


def main():
    print("=" * 60)
    print("🎮 TETRIS RL TRAINING DIAGNOSTIC TOOL")
    print("=" * 60)
    
    diagnostic = TrainingDiagnostic()
    
    # Find files
    files = diagnostic.find_files()
    
    # Analyze checkpoint if available
    if files['checkpoints']:
        latest_checkpoint = max(files['checkpoints'], key=os.path.getmtime)
        diagnostic.analyze_checkpoint(latest_checkpoint)
    
    # Analyze logs if available
    if files['logs']:
        # Prefer CSV files
        csv_logs = [f for f in files['logs'] if f.endswith('.csv')]
        if csv_logs:
            latest_log = max(csv_logs, key=os.path.getmtime)
            diagnostic.analyze_logs(latest_log)
    
    # Generate diagnosis
    diagnostic.diagnose_issues()
    
    print("\n" + "=" * 60)
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()