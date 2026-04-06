```bash
python3 train.py \
    --env_name cage \
    --exp_name cage65_N40 \
    --N 40 \
    --k 6 \
    --encoding_tokens single_integer \
    --max_len 350 \
    --gensize 1500000 \
    --pop_size 10000 \
    --num_samples_from_model 5000 \
    --max_epochs 1000 \
    --max_steps 500 \
    --temperature 0.8 \
    --inc_temp 0.1 \
    --always_search true \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 256 \
    --batch_size 128
```
