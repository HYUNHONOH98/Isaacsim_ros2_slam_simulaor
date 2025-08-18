#!/bin/bash

# Script name: run_eetrack_play.sh

# Initialize variables
HEADLESS_FLAG=""
LOAD_RUN=""
NUM_ENVS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless)
            HEADLESS_FLAG="--headless"
            shift
            ;;
        --load_run)
            if [[ -n "$2" ]]; then
                LOAD_RUN="--load_run $2"
                shift 2
            else
                echo "Error: --run_name requires an argument."
                exit 1
            fi
            ;;
        --test_type)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                TEST_TYPE="--test_type $2"
                shift 2
            fi
            ;;
        --num_envs)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                NUM_ENVS="--num_envs $2"
                shift 2
            else
                echo "Error: --num_envs requires a valid integer."
                exit 1
            fi
            ;;
        *)
            OTHER_ARGS+="$1 "
            shift
            ;;
    esac
done


echo "
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task yt-play \
    --experiment_name g1_walk \
    --video --video_length 406 \
    $HEADLESS_FLAG \
    $LOAD_RUN \
    $TEST_TYPE \
    $NUM_ENVS $OTHER_ARGS
"

# Run the Python script with the appropriate flags
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task locomotion-play \
    --experiment_name g1_walk \
    --video --video_length 406 \
    $HEADLESS_FLAG \
    $LOAD_RUN \
    $TEST_TYPE \
    $NUM_ENVS $OTHER_ARGS

