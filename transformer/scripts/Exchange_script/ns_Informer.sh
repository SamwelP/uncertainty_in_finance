# Reproduced code from https://github.com/thuml/Nonstationary_Transformers (Samwel Portelli <samwel.portelli.18@um.edu.mt>)

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model ns_Informer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --gpu 0 \
  --des 'Exp_h16_l2' \
  --p_hidden_dims 16 16 \
  --p_hidden_layers 2 \
  --itr 1 & 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model ns_Informer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --gpu 1 \
  --des 'Exp_h16_l2' \
  --p_hidden_dims 16 16 \
  --p_hidden_layers 2 \
  --itr 1 & 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model ns_Informer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --gpu 2 \
  --des 'Exp_h64_l1' \
  --p_hidden_dims 64 \
  --p_hidden_layers 1 \
  --itr 1 & 


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model ns_Informer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --gpu 3 \
  --des 'Exp_h64_l2' \
  --p_hidden_dims 64 64 \
  --p_hidden_layers 2 \
  --itr 1 & 
