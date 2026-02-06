@echo off
REM Windows Batch Script for Landslide Detection Training and Inference
REM Usage: run_experiment.bat [train|predict|visualize|full|baseline]

setlocal enabledelayedexpansion

REM Configuration
set DATA_DIR=.\data
set CHECKPOINT_DIR=.\checkpoints
set PREDICTION_DIR=.\predictions
set VISUALIZATION_DIR=.\visualizations
set MODEL_TYPE=attention_unet
set BATCH_SIZE=16
set NUM_EPOCHS=100
set GPU_ID=0

REM Check command
if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="train" goto train
if "%1"=="predict" goto predict
if "%1"=="test" goto test_predict
if "%1"=="visualize" goto visualize
if "%1"=="full" goto full_pipeline
if "%1"=="baseline" goto baseline
goto help

:train
echo ================================================
echo Starting Model Training
echo ================================================

REM Check if data exists
if not exist "%DATA_DIR%\TrainData" (
    echo [ERROR] Training data not found at %DATA_DIR%\TrainData
    echo Please download the Landslide4Sense dataset first
    exit /b 1
)

if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

python src/train.py ^
    --data_dir "%DATA_DIR%" ^
    --save_dir "%CHECKPOINT_DIR%" ^
    --model_type %MODEL_TYPE% ^
    --num_epochs %NUM_EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr 0.001 ^
    --optimizer adamw ^
    --scheduler plateau ^
    --landslide_weight 2.5 ^
    --ce_weight 0.5 ^
    --dice_weight 0.5 ^
    --use_amp ^
    --gpu_id %GPU_ID% ^
    --num_workers 4 ^
    --seed 42

echo [INFO] Training completed! Best model saved to %CHECKPOINT_DIR%\best_model.pth
goto end

:predict
echo ================================================
echo Generating Predictions on Validation Set
echo ================================================

if not exist "%CHECKPOINT_DIR%\best_model.pth" (
    echo [ERROR] No trained model found at %CHECKPOINT_DIR%\best_model.pth
    echo Please train a model first using: run_experiment.bat train
    exit /b 1
)

if not exist "%PREDICTION_DIR%\valid" mkdir "%PREDICTION_DIR%\valid"

python src/predict.py ^
    --data_dir "%DATA_DIR%" ^
    --split valid ^
    --checkpoint "%CHECKPOINT_DIR%\best_model.pth" ^
    --model_type %MODEL_TYPE% ^
    --output_dir "%PREDICTION_DIR%\valid" ^
    --find_threshold ^
    --use_tta ^
    --batch_size 16 ^
    --gpu_id %GPU_ID% ^
    --create_zip

echo [INFO] Predictions saved to %PREDICTION_DIR%\valid
goto end

:test_predict
echo ================================================
echo Generating Predictions on Test Set
echo ================================================

if not exist "%CHECKPOINT_DIR%\best_model.pth" (
    echo [ERROR] No trained model found
    exit /b 1
)

if not exist "%PREDICTION_DIR%\test" mkdir "%PREDICTION_DIR%\test"

python src/predict.py ^
    --data_dir "%DATA_DIR%" ^
    --split test ^
    --checkpoint "%CHECKPOINT_DIR%\best_model.pth" ^
    --model_type %MODEL_TYPE% ^
    --output_dir "%PREDICTION_DIR%\test" ^
    --threshold 0.40 ^
    --use_tta ^
    --batch_size 16 ^
    --gpu_id %GPU_ID% ^
    --create_zip

echo [INFO] Test predictions saved to %PREDICTION_DIR%\test
goto end

:visualize
echo ================================================
echo Creating Visualizations
echo ================================================

if not exist "%PREDICTION_DIR%\valid" (
    echo [ERROR] No predictions found. Please run prediction first.
    exit /b 1
)

if not exist "%VISUALIZATION_DIR%" mkdir "%VISUALIZATION_DIR%"

python src/visualize.py ^
    --image_dir "%DATA_DIR%\TrainData\img" ^
    --mask_dir "%DATA_DIR%\TrainData\mask" ^
    --pred_dir "%PREDICTION_DIR%\valid" ^
    --output_dir "%VISUALIZATION_DIR%" ^
    --num_samples 10

echo [INFO] Visualizations saved to %VISUALIZATION_DIR%
goto end

:full_pipeline
echo ================================================
echo Running Full Pipeline
echo ================================================

call :train
if errorlevel 1 exit /b 1

call :predict
if errorlevel 1 exit /b 1

call :visualize
if errorlevel 1 exit /b 1

echo ================================================
echo Full Pipeline Completed Successfully!
echo ================================================
echo Results summary:
echo   - Model: %CHECKPOINT_DIR%\best_model.pth
echo   - Predictions: %PREDICTION_DIR%\
echo   - Visualizations: %VISUALIZATION_DIR%\
goto end

:baseline
echo ================================================
echo Running Quick Baseline Experiment
echo ================================================

set NUM_EPOCHS=20

echo [INFO] Training baseline model for %NUM_EPOCHS% epochs

if not exist "%DATA_DIR%\TrainData" (
    echo [ERROR] Training data not found
    exit /b 1
)

if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

python src/train.py ^
    --data_dir "%DATA_DIR%" ^
    --save_dir "%CHECKPOINT_DIR%" ^
    --model_type %MODEL_TYPE% ^
    --num_epochs %NUM_EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr 0.001 ^
    --optimizer adamw ^
    --scheduler plateau ^
    --landslide_weight 2.5 ^
    --use_amp ^
    --gpu_id %GPU_ID%

call :predict

echo ================================================
echo Baseline Experiment Completed!
echo ================================================
goto end

:help
echo Usage: run_experiment.bat [command]
echo.
echo Commands:
echo   train       - Train the model
echo   predict     - Generate predictions on validation set
echo   test        - Generate predictions on test set
echo   visualize   - Create visualizations of results
echo   full        - Run complete pipeline (train + predict + visualize)
echo   baseline    - Quick baseline experiment (20 epochs)
echo   help        - Show this help message
echo.
echo Configuration (edit script to modify):
echo   DATA_DIR: %DATA_DIR%
echo   MODEL_TYPE: %MODEL_TYPE%
echo   BATCH_SIZE: %BATCH_SIZE%
echo   NUM_EPOCHS: %NUM_EPOCHS%
echo   GPU_ID: %GPU_ID%
goto end

:end
endlocal