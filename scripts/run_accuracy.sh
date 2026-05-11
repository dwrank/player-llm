#!/bin/bash
python3 tests/test_accuracy.py --variants gguf-q8 --n-samples 1000 2>&1 | tee /tmp/accuracy_1000_q8.log
