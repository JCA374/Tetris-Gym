# src/env.py
"""
Simplified environment module for Tetris Gymnasium
Main environment creation logic is now in config.py
"""

from config import make_env, test_environment

# Re-export the main functions for backward compatibility
__all__ = ['make_env', 'test_environment']

# This file now serves as a simple import wrapper
# All environment logic has been moved to config.py for better organization
