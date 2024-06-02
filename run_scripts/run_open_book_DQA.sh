accelerate launch --main_process_port 29502 \
    --config_file recipes/accelerate_configs/deepspeed_zero3_sft.yaml \
    my_scripts/run_open_book.py \
    recipes/run_AL/open_book/config_full_DQA.yaml