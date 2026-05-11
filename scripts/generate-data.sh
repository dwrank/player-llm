#!/bin/bash
python3 parse_phh.py \
      --mode sft \
      --input /data/poker/phh-dataset/data/pluribus \
      --output ../data/sft_pluribus.jsonl
