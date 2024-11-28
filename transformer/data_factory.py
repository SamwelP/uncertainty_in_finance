# Adapted code from https://github.com/thuml/Nonstationary_Transformers (Samwel Portelli <samwel.portelli.18@um.edu.mt>)

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Customv1
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'not_batched_train':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq    
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    if flag == 'not_batched_train':
        flag = 'train'
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    
    scaler = data_set.get_scaler()
    scaled_data_x = data_set.get_data_x()
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader, scaler, scaled_data_x

# Added the index for EnBPI
def data_providerv1(args, flag, index_array):
    #Data = data_dict[args.data]
    Data = Dataset_Customv1
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'not_batched_train':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if flag == 'not_batched_train':
        flag = 'train'

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        index_array=index_array
    )
    
    scaler = data_set.get_scaler()
    scaled_data_x = data_set.get_data_x()
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader, scaler, scaled_data_x
    
