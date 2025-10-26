# monitor_training.py
"""
Monitor Tetris training progress in real-time
Usage: python monitor_training.py [log_file]
"""

import sys
import time
import re
from pathlib import Path

def parse_episode_line(line):
    """Extract metrics from episode log line"""
    # Episode 100 | Lines: 5 (Total: 123) | Reward: 456.7 (Avg: 234.5) | Steps: 89 (Avg: 76.2) | Lines/Ep: 1.23 | ε: 0.950 | Shaping: YES
    episode_match = re.search(r'Episode\s+(\d+)', line)
    lines_match = re.search(r'Lines:\s+(\d+)', line)
    total_lines_match = re.search(r'Total:\s+(\d+)', line)
    reward_match = re.search(r'Reward:\s+([\d.]+)', line)
    avg_reward_match = re.search(r'Avg:\s+([\d.]+)', line)
    epsilon_match = re.search(r'ε:\s+([\d.]+)', line)
    
    if episode_match:
        return {
            'episode': int(episode_match.group(1)),
            'lines': int(lines_match.group(1)) if lines_match else 0,
            'total_lines': int(total_lines_match.group(1)) if total_lines_match else 0,
            'reward': float(reward_match.group(1)) if reward_match else 0,
            'avg_reward': float(avg_reward_match.group(1)) if avg_reward_match else 0,
            'epsilon': float(epsilon_match.group(1)) if epsilon_match else 0
        }
    return None

def monitor_log(log_file, interval=5):
    """Monitor training log and show progress"""
    print(f"📊 Monitoring: {log_file}")
    print(f"🔄 Refreshing every {interval} seconds (Ctrl+C to stop)\n")
    
    last_size = 0
    last_episode = None
    episode_history = []
    
    try:
        while True:
            try:
                with open(log_file, 'r') as f:
                    # Read all lines
                    lines = f.readlines()
                    
                    # Parse episode lines
                    for line in lines:
                        if 'Episode' in line and '|' in line:
                            metrics = parse_episode_line(line)
                            if metrics:
                                episode_history.append(metrics)
                    
                    # Show summary
                    print("\033[H\033[J")  # Clear screen
                    print(f"📊 TRAINING PROGRESS - {time.strftime('%H:%M:%S')}")
                    print("=" * 70)
                    
                    if episode_history:
                        latest = episode_history[-1]
                        print(f"🎯 Current Episode: {latest['episode']}/10000")
                        print(f"📈 Progress: {latest['episode']/100:.1f}%")
                        print(f"🎮 Total Lines Cleared: {latest['total_lines']}")
                        print(f"💰 Latest Reward: {latest['reward']:.1f}")
                        print(f"📊 Average Reward: {latest['avg_reward']:.1f}")
                        print(f"🎲 Epsilon: {latest['epsilon']:.4f}")
                        
                        # Calculate trends
                        if len(episode_history) >= 20:
                            recent_20 = episode_history[-20:]
                            avg_lines = sum(e['lines'] for e in recent_20) / len(recent_20)
                            avg_reward_recent = sum(e['reward'] for e in recent_20) / len(recent_20)
                            
                            print("\n📈 Recent Performance (last 20 episodes):")
                            print(f"   Lines/Episode: {avg_lines:.2f}")
                            print(f"   Avg Reward: {avg_reward_recent:.1f}")
                        
                        # Show last few episodes
                        print("\n📋 Last 5 Episodes:")
                        print("   Ep    | Lines | Reward  | ε")
                        print("   " + "-" * 40)
                        for ep in episode_history[-5:]:
                            print(f"   {ep['episode']:4d}  | {ep['lines']:5d} | {ep['reward']:7.1f} | {ep['epsilon']:.4f}")
                    else:
                        print("⏳ Waiting for training to start...")
                        print(f"📝 Log file has {len(lines)} lines")
                        
                        # Show last few lines
                        if lines:
                            print("\n📄 Last 5 lines:")
                            for line in lines[-5:]:
                                print(f"   {line.rstrip()}")
                    
                    print("\n" + "=" * 70)
                    print(f"💡 Tip: Training logs every {100} episodes, saves checkpoints every {500}")
                    
            except FileNotFoundError:
                print(f"⏳ Waiting for log file: {log_file}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        if episode_history:
            print(f"\n📊 Final Stats:")
            print(f"   Episodes completed: {episode_history[-1]['episode']}")
            print(f"   Total lines: {episode_history[-1]['total_lines']}")
            print(f"   Latest avg reward: {episode_history[-1]['avg_reward']:.1f}")

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else "training_overnight.log"
    
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        print(f"💡 Usage: python monitor_training.py [log_file]")
        print(f"💡 Example: python monitor_training.py training_overnight.log")
        return
    
    monitor_log(log_file, interval=5)

if __name__ == "__main__":
    main()
