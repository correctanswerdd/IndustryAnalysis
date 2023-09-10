#!/bin/bash

python main_10k.py --dir_logs 2013_hntm_20000full_out/ --full "full" --extreme 20000 --num_topics "48 32 24 11" --layer_sizes "1024 512 256 128" --embedding_sizes "1600 1200 800 400" --year 2013 --num_epochs 150

python main_10k.py --dir_logs 2014_hntm_20000full_out/ --full "full" --extreme 20000 --num_topics "48 32 24 11" --layer_sizes "1024 512 256 128" --embedding_sizes "1600 1200 800 400" --year 2014 --num_epochs 150

python main_10k.py --dir_logs 2015_hntm_20000full_out/ --full "full" --extreme 20000 --num_topics "48 32 24 11" --layer_sizes "1024 512 256 128" --embedding_sizes "1600 1200 800 400" --year 2015 --num_epochs 150

python main_10k.py --dir_logs 2016_hntm_20000full_out/ --full "full" --extreme 20000 --num_topics "48 32 24 11" --layer_sizes "1024 512 256 128" --embedding_sizes "1600 1200 800 400" --year 2016 --num_epochs 150

python main_10k.py --dir_logs 2017_hntm_20000full_out/ --full "full" --extreme 20000 --num_topics "48 32 24 11" --layer_sizes "1024 512 256 128" --embedding_sizes "1600 1200 800 400" --year 2017 --num_epochs 150

python main_10k.py --dir_logs 2018_hntm_20000full_out/ --full "full" --extreme 20000 --num_topics "48 32 24 11" --layer_sizes "1024 512 256 128" --embedding_sizes "1600 1200 800 400" --year 2018 --num_epochs 150