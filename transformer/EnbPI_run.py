# Adapted code from https://github.com/thuml/Nonstationary_Transformers (Samwel Portelli <samwel.portelli.18@um.edu.mt>)

import argparse
import torch
import copy
import csv
import pandas as pd
from exp.EnbPI_exp_main import Exp_Main
from deel.puncc.api.prediction import BasePredictor
from regression import EnbPI
from deel.puncc.regression import SplitCP
from deel.puncc.metrics import regression_ace
from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness
from data_provider.data_factory import data_providerv1
import random
import numpy as np
import matplotlib.pyplot as plt

def direction_accuracy(actual, predicted):
    """
    Calculate the direction accuracy between actual and predicted values.

    Parameters:
    actual (list): List of actual values (time series).
    predicted (list): List of predicted values (time series).

    Returns:
    float: Direction accuracy in percentage.
    """
    # Ensure the lengths of actual and predicted are the same
    if len(actual) != len(predicted):
        raise ValueError("Input lists must have the same length.")

    # Calculate differences between consecutive values
    actual_diff = [actual[i] - actual[i-1] for i in range(1, len(actual))]
    predicted_diff = [predicted[i] - predicted[i-1] for i in range(1, len(predicted))]

    # Count the number of correct predictions
    correct_predictions = sum((a * p) > 0 for a, p in zip(actual_diff, predicted_diff))

    # Calculate accuracy as a percentage
    accuracy_percentage = (correct_predictions / len(actual_diff)) * 100.0

    return accuracy_percentage

# Testing a simple CP technique

def calculate_quantile(scores_calib, delta):
    # Calculate the quantile value based on delta and non-conformity scores
    which_quantile = np.ceil((delta)*(n_calib + 1))/n_calib
    return np.quantile(scores_calib, which_quantile, method='lower')

def calibrate(setting, X_calib, delta, n_calib):
    # Calibrate the conformalizer to calculate q_hat
    y_calib_pred, y_calib_true, x_test = exp.custom_test(setting, X_calib)
    y_calib_pred = y_calib_pred[:, -1, 0]
    y_calib_true = y_calib_true[:, -1, 0]
    scores_calib = non_conformity_func(y_calib_pred, y_calib_true)
    q_hat = calculate_quantile(scores_calib, delta)
    
    return q_hat

def conformal_predict(X, setting, q_hat):
    # returns the predicted interval
    y_pred, y_true, x_test = exp.custom_test(setting, X)
    y_pred = y_pred[:, -1, 0]
    y_true = y_true[:, -1, 0]
    y_lower, y_upper = y_pred - q_hat, y_pred + q_hat
    
    plt.figure()
    # Plot the mean values as a line
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Prediction', color='black')
    
    # Fill the area between the minimum and maximum values to represent confidence bands
    plt.fill_between(range(len(y_lower)), y_lower.squeeze(), y_upper.squeeze(), alpha=0.3, color='gray', label='Confidence Band')
    
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    
    
    plt.savefig('plot_with_CP.png')
    
    # Show the plot
    plt.show()
    return y_lower, y_pred, y_upper


def non_conformity_func(y, y_hat):
  return np.abs(y - y_hat)
     

parser = argparse.ArgumentParser(description='Non-stationary Transformers for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Transformer',
                    help='model name, options: [ns_Transformer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh2', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed) 
np.random.seed(fix_seed)

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)

print('Args in experiment:')
print(args)

Exp = Exp_Main

class TransformerPredictor(BasePredictor):
    def fit(self, X, b, y=None, **kwargs):
                
        print('PUNCC is training a model...')
        print('Input Dataset Shape:')
        print(X.shape)
        
                
        self.model, _, _ = exp.trainv1(X)
        print(self.model)
        
        torch.save(self.model.state_dict(), str(b)+'.pth')
        
        return self.model
        
        
    
    def predict(self, X, b, **kwargs):    
        
        print('PUNCC is testing a model...')
        print('Input Dataset Shape:')
        print(X.shape)
        
        
        
        model_path = r'/content/drive/MyDrive/masters back up/Nonstationary-Transfomers/Nonstationary_Transformers-main/Nonstationary_Transformers-main/pre_trained_model_0/'+str(b)+'.pth'
        self.model.load_state_dict(torch.load(model_path))

        
        if len(X)>5000: # If not integers (i.e not indexes)
            print('Integer mode')
            X = np.arange(0,5215)
            
            y_pred, y_true_int_mode, _ = exp.custom_testv6(self.model, X) # (always same order according to github code ON TRAIN DATA (As re. by EnbPI)
            
        else:
            print('Prediction mode')
            
            y_pred = exp.custom_testv5(self.model, X)
                               
        
        
        y_pred = np.array(y_pred).flatten()
        print('y_pred_shape')
        print(y_pred.shape)
        
        
        return y_pred
    
    


if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        model, x_trains, y_trains = exp.train(setting)
        y_pred, y_true_int_mode, _ = exp.custom_testv6(model, np.arange(0,5215))
        
        ### TESTING 
        y_pred, y_true_full, X_values = exp.custom_test(setting, np.arange(0,760))
        plt.figure()
        
        # Plot the mean values as a line
        plt.plot(np.array(y_true_full).flatten(), label='Actual', color='blue')
        plt.plot(np.array(y_pred).flatten(), label='Prediction', color='black')
        
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        
        
        plt.savefig('plot_preds_test.png')
               
        
        print('>>>>>>>CUSTOM testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
       
        transformer_predictor = TransformerPredictor(model, is_trained=False)
        
        enbpi = EnbPI(
            transformer_predictor,
            B=30,
            agg_func_loo=np.mean,
            random_state=0,
        )
        
        
        
        print('Obtaining conformity scores for the calibration data...')
        
        
        y_pred_train, y_train, X_train = exp.custom_testv3(list(range(0, 5216)))
        
        print('EnbPI is fitting...')
        
        
        
        print('Lenght of y_true_int_mode: '+str(len(y_true_int_mode)))
        #enbpi.fit(np.arange(0,5215),y_true_int_mode)#, y_train)#5184 >>> UNCOMMENT TO GENERATE NEW MODELS
        enbpi.fakefit(np.arange(0,5215),y_true_int_mode) #>>>> UNCOMMENT TO NOT GENEATE NEW MODELS
        
        print('EnbPI is done fitting...')
        
        
        test_flag = True
        
        if test_flag:
        
            print('Predicting confidence levels on the test data...')
            
            
            
            y_pred, y_true_full, X_values = exp.custom_test(setting, np.arange(0,760))
            
            
           # Check if y_pred is already a NumPy array
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            # Check if y_true_full is already a NumPy array
            if not isinstance(y_true_full, np.ndarray):
                y_true_full = np.array(y_true_full)
            
            y_true = y_true_full
            
            
            
            alphas = [0.01, 0.05, 0.1, 0.15]
            ss = [100, 50, 25, 20, 10, 5]
            
            for s in ss:
                
        
                try:
                    
                    data_filename, y_pred_from_PUNCC = enbpi.predict(np.arange(0, 760), alpha=alphas, y_true=y_true, s=s) 
                    print('DATA IS SAVED IN: ' + data_filename)
                    actual_pred_data = {
                                         'y_true': y_true,
                                        'y_pred_from_PUNCC': y_pred_from_PUNCC}
                                        
                    # Create a DataFrame
                    df = pd.DataFrame(actual_pred_data)
                    
                    # Specify the CSV file name
                    preds_actual_csv_file_name = 'y_true_and_pred_s_'+str(s)+'.csv'
                    
                    # Write to the CSV file
                    df.to_csv(preds_actual_csv_file_name, index=False)
                    
                    print(f"Data saved to {preds_actual_csv_file_name}")
                    
                except Exception as e:
                    print(f"An error occurred: {e}")
            
                    

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
