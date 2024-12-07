#!/bin/bash

# Function to kill MLflow UI on exit
# cleanup() {
#     echo "Stopping MLflow UI..."
#     kill $MLFLOW_PID
#     echo "MLflow UI stopped."
# }

# Trap SIGINT and SIGTERM to cleanup properly
# trap cleanup SIGINT SIGTERM

# Start MLflow UI in the background
# mlflow ui &
# MLFLOW_PID=$!

# Capture the current working directory
SCRIPT_DIR=$(dirname "$0")

# Record the start time for the entire script
total_start_time=$(date +%s)

# Change directory to the configs directory
cd "$SCRIPT_DIR"/../configs

# Loop through all .yaml files in the configs directory
for config_file in *.yaml; do
    echo "=========>Training with configuration file: $config_file"
    # Record the start time for the current config file
    start_time=$(date +%s)

    # Change directory to src where the Python script is expected to run
    cd "$SCRIPT_DIR"/../src

    # Run the training script with the current config file
    python train_CIFAR.py "../configs/$config_file"
    
    # Record the end time for the current config file
    end_time=$(date +%s)
    
    # Calculate the elapsed time for the current config file
    elapsed_time=$((end_time - start_time))
    
    # Print the elapsed time for the current config file
    echo "Elapsed time for $config_file: $elapsed_time seconds"
    echo "======================================================================"
    # Return to configs directory to prepare for the next iteration
    cd "$SCRIPT_DIR"/../configs
done

# Instructions for the user at the end of the script
# echo "MLflow UI is still running in the background."
# echo "To access MLflow UI, open http://localhost:5000 in your web browser."
# echo "To stop MLflow UI, you can use 'kill \$(pgrep -f 'mlflow')'."

# Record the end time for the entire script
total_end_time=$(date +%s)

# Calculate the total elapsed time for the script
total_elapsed_time=$((total_end_time - total_start_time))

# Print the total elapsed time
echo "Total elapsed time for the script: $total_elapsed_time seconds"

# Echo the end time of the whole script
end_time_formatted=$(date '+%Y-%m-%d %H:%M:%S')
echo "======================================================================"
echo "Configured traning jobs finished at: $end_time_formatted"

# Return to the original directory
cd "$SCRIPT_DIR"