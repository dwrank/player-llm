#!/bin/bash
watch -n 0.5 nvidia-smi --query-gpu=memory.used --format=csv,noheader
