# Adapted code from https://github.com/thuml/Nonstationary_Transformers (Samwel Portelli <samwel.portelli.18@um.edu.mt>)

import argparse
import torch
from exp.GARCH_exp_main import Exp_Main
import random
import numpy as np
import arch
import scipy.stats as stats
import matplotlib.pyplot as plt
from deel.puncc.metrics import regression_ace
from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness
import pandas as pd
import csv 

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
    predicted_diff = [predicted[i] - actual[i-1] for i in range(1, len(predicted))]

    # Count the number of correct predictions
    correct_predictions = sum((a * p) > 0 for a, p in zip(actual_diff, predicted_diff))

    # Calculate accuracy as a percentage
    accuracy_percentage = (correct_predictions / len(actual_diff)) * 100.0

    return accuracy_percentage

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
torch.manual_seed(fix_seed) ## SAMWEL MODEL FOR DEEP ENSEMBLE
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

lags=8
targets=['0', '1', '2', '3', '4', '5', '6', 'OT']
p=1
q=1

for target in targets:

    Exp = Exp_Main
    
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
    
            args.target = target
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            model, y_pred_train, y_train = exp.train(setting)
            
            # Getting the full train dataset
            y_pred_train, y_train = exp.traintest(setting)
            
            print('y_pred_train shape: ' +str(y_pred_train.shape))
            print('y_train shape: ' +str(y_train.shape))
            
            plt.figure()
            
            # Plot the mean values as a line
            plt.plot(y_train.flatten(), label='Actual', color='blue')
            plt.plot(y_pred_train.flatten(), label='Train Prediction from Transformer', color='black')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            
            
            plt.savefig('Traindata.png')
            
            using_ln = False
            
            # Training the GARCH
            if not using_ln:
                error = y_train - y_pred_train
                error = error[-lags:]
            else:
                error = np.exp(y_train) - np.exp(y_pred_train)
            
            plt.figure()
            
            # Plot the mean values as a line
            plt.plot(error.flatten(), label='Error', color='blue')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            plt.savefig('Error.png')
            
            print('Lenght of garch fitting array: ' + str(len(error)))
            
            garch = arch.arch_model(error, p=p, q=q, vol="Garch", dist="Normal")
            garch_model = garch.fit(update_freq=1, disp=False)
    
    
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            valpreds, valtrues = exp.valtest(setting) # Obtaining the validation datset preds and trues
            testpreds, testtrues = exp.test(setting) # Obtaining the test datset preds and trues
            
            # concatenating the preds and trues
            preds = np.concatenate((valpreds, testpreds))
            trues = np.concatenate((valtrues, testtrues))
            
            if using_ln:
                preds = np.exp(preds)
                trues = np.exp(trues)
            
            print('GARCH Fitting')
            
            alphas = [0.01, 0.05, 0.1, 0.15]
            
            for alpha in alphas:
                
                print('Testing alpha: '+str(alpha))
                
                garch_upper, garch_lower = [], []
                
                # Getting the confidence from the GARCH
                for i in range(len(preds)):
                    garch_forecast = garch_model.forecast(horizon=1)
                    garch_variance = garch_forecast.variance['h.1'].iloc[-1]
                    
                    garch_upper.append(preds[i] + stats.norm.ppf(1 - alpha / 2)*np.sqrt(garch_variance))
                    garch_lower.append(preds[i] - stats.norm.ppf(1 - alpha / 2)*np.sqrt(garch_variance))
                    
                    if i == 0:
                        updated_errors = np.concatenate((error[1:], np.array((trues[i] - preds[i])).reshape(1)))
                    else:
                        updated_errors = np.concatenate((updated_errors[1:], np.array((trues[i] - preds[i])).reshape(1)))
                        
                    garch = arch.arch_model(updated_errors, p=p, q=q, vol="Garch", dist="Normal")
                    garch_model = garch.fit(update_freq=1, disp=False)
                
                print('garch_upper shape: ' +str(len(garch_upper)))
                print('garch_lower shape: ' +str(len(garch_lower)))
                print('trues shape: ' +str(trues.shape))
                print('preds shape: ' +str(preds.shape))
                
                val_garch_upper = garch_upper[:len(valtrues)]
                val_garch_lower = garch_lower[:len(valtrues)]
                test_garch_upper = garch_upper[len(valtrues):]
                test_garch_lower = garch_lower[len(valtrues):]
                
                # VALIDATION <------------------------------------------------------------------------------------
                plt.figure()
                
                # Plot the mean values as a line
                plt.plot(valtrues.flatten(), label='Actual', color='blue')
                plt.plot(valpreds.flatten(), label='Prediction from Transformer', color='black')
                
                # Fill the area between the minimum and maximum values to represent confidence bands
                plt.fill_between(range(len(val_garch_lower)), np.array(val_garch_lower), np.array(val_garch_upper), alpha=0.3, color='gray', label='Confidence Band')
                
                plt.xlabel('Index')
                plt.ylabel('Values')
                plt.legend()
                plt.grid(True)
                
                
                plt.savefig('val_plot_with_CP_GARCH_alpha_'+str((1-alpha)*100)+'_'+target+'.png')
                
                #--------------------- Calculating the metrics of the results
                ace = regression_ace(valtrues.flatten(), np.array(val_garch_lower), np.array(val_garch_upper), alpha)
                mean_coverage = regression_mean_coverage(valtrues.flatten(), np.array(val_garch_lower), np.array(val_garch_upper))
                average_width = regression_sharpness(np.array(val_garch_lower), np.array(val_garch_upper))
                direction_accuracy_value = direction_accuracy(valtrues.flatten(),valpreds.flatten())
                
                print('The average coverage error is: '+str(ace))
                print('The mean coverage: '+str(mean_coverage))
                print('The PI average width: '+str(average_width))
                print('The direction_accuracy_value is: '+str(direction_accuracy_value))
        
                # Create a list with the values
                cp_metric_data = [['ace', ace], ['mean_coverage', mean_coverage], ['average_width', average_width], ['direction_accuracy_value', direction_accuracy_value]]
                
                # Specify the CSV file name
                metrics_csv_file_name = 'val_garch_transformer_confidence_metrics_alp_'+str(int((1-alpha)*100))+'_'+target+'.csv'
                
                # Write to the CSV file
                with open(metrics_csv_file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Write the header
                    writer.writerow(['Metric', 'Value'])
                    # Write the data
                    writer.writerows(cp_metric_data)
                
                print(f"Metrics saved to {metrics_csv_file_name}")
                
                # Create a dictionary with the arrays
                actual_pred_data = {
                    'y_true': valtrues.flatten(),
                    'y_pred_from_PUNCC': valpreds.flatten(),
                    'y_lower': np.array(val_garch_lower),
                    'y_upper': np.array(val_garch_upper)
                }
                
                # Create a DataFrame
                df = pd.DataFrame(actual_pred_data)
                
                # Specify the CSV file name
                preds_actual_csv_file_name = 'val_garch_transforemer_y_true_and_pred_alp_'+str(int((1-alpha)*100))+'_'+target+'.csv'
                
                # Write to the CSV file
                df.to_csv(preds_actual_csv_file_name, index=False)
                
                print(f"Data saved to {preds_actual_csv_file_name}")
                
                # TEST <------------------------------------------------------------------------------------
                plt.figure()
                
                # Plot the mean values as a line
                plt.plot(testtrues.flatten(), label='Actual', color='blue')
                plt.plot(testpreds.flatten(), label='Prediction from Transformer', color='black')
                
                # Fill the area between the minimum and maximum values to represent confidence bands
                plt.fill_between(range(len(test_garch_lower)), np.array(test_garch_lower), np.array(test_garch_upper), alpha=0.3, color='gray', label='Confidence Band')
                
                plt.xlabel('Index')
                plt.ylabel('Values')
                plt.legend()
                plt.grid(True)
                
                
                plt.savefig('plot_with_CP_GARCH_alpha_'+str((1-alpha)*100)+'_'+target+'.png')
                
                #--------------------- Calculating the metrics of the results
                ace = regression_ace(testtrues.flatten(), np.array(test_garch_lower), np.array(test_garch_upper), alpha)
                mean_coverage = regression_mean_coverage(testtrues.flatten(), np.array(test_garch_lower), np.array(test_garch_upper))
                average_width = regression_sharpness(np.array(test_garch_lower), np.array(test_garch_upper))
                direction_accuracy_value = direction_accuracy(testtrues.flatten(),testpreds.flatten())
                
                print('The average coverage error is: '+str(ace))
                print('The mean coverage: '+str(mean_coverage))
                print('The PI average width: '+str(average_width))
                print('The direction_accuracy_value is: '+str(direction_accuracy_value))
        
                # Create a list with the values
                cp_metric_data = [['ace', ace], ['mean_coverage', mean_coverage], ['average_width', average_width], ['direction_accuracy_value', direction_accuracy_value]]
                
                # Specify the CSV file name
                metrics_csv_file_name = 'garch_transformer_confidence_metrics_alp_'+str(int((1-alpha)*100))+'_'+target+'.csv'
                
                # Write to the CSV file
                with open(metrics_csv_file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Write the header
                    writer.writerow(['Metric', 'Value'])
                    # Write the data
                    writer.writerows(cp_metric_data)
                
                print(f"Metrics saved to {metrics_csv_file_name}")
                
                # Create a dictionary with the arrays
                actual_pred_data = {
                    'y_true': testtrues.flatten(),
                    'y_pred_from_PUNCC': testpreds.flatten(),
                    'y_lower': np.array(test_garch_lower),
                    'y_upper': np.array(test_garch_upper)
                }
                
                # Create a DataFrame
                df = pd.DataFrame(actual_pred_data)
                
                # Specify the CSV file name
                preds_actual_csv_file_name = 'garch_transforemer_y_true_and_pred_alp_'+str(int((1-alpha)*100))+'_'+target+'.csv'
                
                # Write to the CSV file
                df.to_csv(preds_actual_csv_file_name, index=False)
                
                print(f"Data saved to {preds_actual_csv_file_name}")
    
    
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
