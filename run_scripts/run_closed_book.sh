accelerate launch --main_process_port 29502 \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    my_scripts/run_closed_book.py \
    recipes/run_AL/closed_book/config_full_en.yaml