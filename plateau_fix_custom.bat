@echo off
REM Customizable Tetris AI Plateau Breaking Strategy - Windows Version

REM Configuration - adjust these numbers as needed
set CURRENT_EPISODE=1000
set PLATEAU_BREAKER_EPISODES=500
set ADDITIONAL_TRAINING=1500
set /a FINAL_EPISODE=%CURRENT_EPISODE% + %ADDITIONAL_TRAINING%

echo 🚨 PLATEAU DETECTED - Applying surgical fixes...
echo Current episode: %CURRENT_EPISODE%
echo Plateau breaker episodes: %PLATEAU_BREAKER_EPISODES%
echo Final training target: %FINAL_EPISODE%
echo.

REM Step 1: Quick diagnosis
echo Step 1: Quick diagnosis...
python tetris_diagnostic.py --episodes 10 --analyze-actions --save-plots
if %errorlevel% neq 0 (
    echo ❌ Diagnostic failed - check your environment setup
    pause
    exit /b 1
)

echo.

REM Step 2: Plateau breaker
echo Step 2: Running plateau breaker...
echo This creates a separate training session to find breakthrough
python break_plateau_train.py --episodes %PLATEAU_BREAKER_EPISODES% --target-lines 20 --dense-freq 2 --epsilon-boost 0.7

REM Check if plateau breaker succeeded
if %errorlevel% equ 0 (
    echo ✅ Plateau breaker completed
) else (
    echo ❌ Plateau breaker failed - trying more aggressive settings
    python break_plateau_train.py --episodes %PLATEAU_BREAKER_EPISODES% --target-lines 15 --dense-freq 1 --epsilon-boost 0.8
)

echo.

REM Step 3: Resume main training
echo Step 3: Resume main training with improvements...
echo Training from episode %CURRENT_EPISODE% to %FINAL_EPISODE%

REM First, backup current model
if exist models\latest_checkpoint.pth (
    copy models\latest_checkpoint.pth models\backup_before_plateau_fix.pth
    echo 📦 Backed up current model to backup_before_plateau_fix.pth
)

python train.py --episodes %FINAL_EPISODE% --resume --reward_shaping simple --experiment_name "post_plateau_breakthrough_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"

echo.
echo 🎉 Training sequence complete!
echo.
echo 📊 Check results in:
echo   - logs\ directory for training curves
echo   - plateau_breaker_results_*.json for breakthrough metrics
echo   - models\ directory for saved checkpoints

pause