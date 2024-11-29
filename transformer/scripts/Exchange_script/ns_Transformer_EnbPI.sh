# Adapted code from https://github.com/thuml/Nonstationary_Transformers (Samwel Portelli <samwel.portelli.18@um.edu.mt>)

python -u "/content/drive/MyDrive/masters back up/Nonstationary-Transfomers/Nonstationary_Transformers-main/Nonstationary_Transformers-main/EnbPI_run.py" \
  --is_training 1 \
  --root_path "/content/drive/MyDrive/masters back up/Nonstationary-Transfomers/dataset/dataset/exchange_rate/" \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model ns_Transformer \
  --data custom \
  --features MS \
  --freq d \
  --target 0 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1 \
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