import utils
import logging
import argparse
import pandas as pd
from models.base import Model
from models import booster, gnn
from data.raw_data_processing import get_data
from configs.configs import split_perc
import inference_saving.save_load_models as slm
from relevant_banks import get_relevant_banks
import pandas as pd
import numpy as np
import itertools
import inspect
import torch
from torch_geometric.data import Data
import data.data_utils as du



# I need to adjust the feature_engineering function, to deal with ports and tds


utils.logger_setup()
parser = utils.get_parser()
args = parser.parse_args()

utils.set_seed(args.seed, True)
model = Model.from_model_type(args)

df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
df_edges = df

raw_data = get_data(df, model.args, split_perc = split_perc)

fr_banks, sr_banks = get_relevant_banks(args)

args.model = 'xgboost'
self = Model.from_model_type(args)
#args.ports = True
#args.tds = True
bank = 0
args.scenario = 'individual_banks'
bank_indices = self.get_indices(raw_data, bank=bank)
train_data, vali_data, test_data = self.get_data(raw_data, bank_indices)


import sys
sys.getsizeof(raw_data)

def get_deep_size(obj, seen=None):
    """Calculate deep memory usage of an object"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_deep_size(v, seen) for v in obj.values()])
        size += sum([get_deep_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_deep_size(i, seen) for i in obj])
    
    return size

def format_bytes(bytes_size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

get_deep_size(raw_data)
format_bytes(4993674593)


fr_sr = fr_banks + sr_banks
test123 = 0
test123 = []
for bank in fr_sr:
    bank_indices = self.get_indices(raw_data, bank=bank)
    train_data, vali_data, test_data = self.get_data(raw_data, bank_indices)
    #test123 += sum(map(get_deep_size, [train_data, vali_data, test_data]))
    test123.append([train_data, vali_data, test_data])


get_deep_size(test123)

format_bytes(22842692)
format_bytes(test123)
format_bytes(1879883736)

get_deep_size(raw_data['graph_data'])
get_deep_size(raw_data.get('regular_data'))
format_bytes(4993664376)


test123, test1234 = self.feature_engineering(train_data, vali_data)


data = train_data


train_data['df'].edge_attr

df = raw_data['graph_data']



train_data, vali_data, test_data = self.get_data(raw_data, bank_indices)
test_data = vali_data


self = Model.from_model_type(args)
train_data, vali_data, test_data = model.get_data(raw_data, bank_indices=None)


data = raw_data[model.args.data_type]['test_data']['df']

bank_indices







# need to check this step:
#     edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
#     edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()
# in the pack_graph_data function

# maybe hear Tarik about the whether nodes omfr the 'future' should be included also for train/vali data, even though 
# they first see a transaction in the vali/test data


df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

# get timestamps and labels
timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

#valid_keys = inspect.signature(split_indices).parameters.keys()
#args = {key: kwargs[key] for key in valid_keys if key in kwargs}

#split_inds, test_perc = split_indices(timestamps, y, **args)
split_inds, test_perc = split_indices(timestamps, y, [0.6, 0.2])

train_indices = np.concatenate(split_inds[0])
vali_indices = np.concatenate(split_inds[1])
test_indices = np.concatenate(split_inds[2])
indices = [train_indices, vali_indices, test_indices]

packed_data = {}






#raw_data = get_data(df, model.args, split_perc = split_perc)











"""


      self.train_data.edge_attr = torch.cat([torch.arange(self.train_data.edge_attr.shape[0]).view(-1, 1), self.train_data.edge_attr], dim=1)

        self.m_param['num_neighbors'] = [5, 5, 0]
        self.m_param['num_neighbors'] = [100, 100, 100]
        self.batch_size = 256
        self.batch_size = 8192
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_loader, test_loader = tgu.get_loaders(self.train_data, self.test_data, self.pred_indices, self.m_param, self.batch_size)

        
        train_loader, test_loader = tgu.get_loaders(Data(x = self.train_data.x, y=self.train_data.y, edge_index=self.train_data.edge_index, edge_attr=self.train_data.edge_attr, timestamps=self.train_data.timestamps), self.test_data, self.pred_indices, self.m_param, self.batch_size)
        batch = next(iter(train_loader))

        batch = next(iter(train_loader))
        inds = self.train_indices.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = train_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
        sum(mask)


"""



