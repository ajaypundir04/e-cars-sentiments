#!/bin/bash

CHECKPOINT_DIR=.llama/checkpoints/Meta-Llama3.1-8B-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun llama_models/scripts/example_chat_completion.py $CHECKPOINT_DIR