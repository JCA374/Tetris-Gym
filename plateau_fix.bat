@echo off
REM Comprehensive Tetris AI Plateau Breaking Strategy - Windows Version

echo 🚨 PLATEAU DETECTED - Applying surgical fixes...
echo Current agent is at episode 1000 - adjusting episode counts accordingly
echo.

REM Step 1: Quick diagnosis (optional but recommended)
echo Step 1: Quick diagnosis...
python tetris_diagnostic.py --episodes 10 --analyze-actions --save-plots
if %errorlevel% neq 0 (
    echo ❌ Diagnostic failed - check your environment setup
    pause
    exit /b 1
)

echo.

REM Step 2: Apply plateau breaker with aggressive settings
echo Step 2: Running plateau breaker...
echo Note: This will run independently to break the plateau
python break_plateau_train.py --episodes 500 --target-lines 20 --dense-freq 2 --epsilon-boost 0.7

echo.

REM Step 3: If plateau breaker succeeds, continue with normal training
echo Step 3: Resume normal training with fixed epsilon...
echo Setting episodes to 2500 (current 1000 + 1500 more)
python train.py --episodes 2500 --resume --reward_shaping simple --experiment_name "post_plateau_breakthrough"

echo.
echo 🎯 Expected timeline:
echo   - Plateau breaker: 50-150 episodes to first line clear
echo   - Normal training: 500+ episodes to consistent performance
echo   - Total time: 2-4 hours depending on hardware

echo.
echo 📊 Success indicators:
echo   - First line clear within 100 episodes
echo   - 5+ lines cleared within 200 episodes
echo   - Consistent 0.5+ lines per episode within 300 episodes

echo.
echo 🔧 If plateau breaker fails, try:
echo   1. Increase --epsilon-boost to 0.8
echo   2. Decrease --dense-freq to 1 (every episode)
echo   3. Consider imitation learning with human demonstrations

echo.
echo 📁 Monitor these files for progress:
echo   - logs\ directory for training progress
echo   - plateau_breaker_results_*.json for breakthrough status
echo   - diagnostics_output\ for analysis plots

echo.
echo ✅ Script completed! Check the output above for any errors.
pause