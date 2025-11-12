def tuning(self, laundering_values_vali):

    # when tuning / training gnn I need to remember to test 4 different seeds


    # --------------------------------
    
    # for boost it would want it inside the hyperparameter loop.
    # however, if spliting and processing several datapoints at once, then
    # maybe it could be avoided, however too much memory usage?
    # So don't do that, find solution to only do once for reg and gnn?
    # will see at booster trees

    # preferable this should be a 'reuseable loop' for all the models
    for bank_id, party in self.parties.items():
        party.prep_data_tuning()

    # probably need to first init hyparameters here

    # need to do this several times for -gnn
    hyperparameters_tuning = self.init_hyperparams()


    # this part here is only needed for gnn --------------------------

    _, scores = self.tuning_loop(hyperparameters_tuning, laundering_values_vali)
    params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
    sample_space = self._get_search_space(top_parameters)

    #init_hp = self.init_hyperparams(sample_space)
    #hyperparameters_tuning = init_hp['hp_list']
    hyperparameters_tuning = self.init_hyperparams(sample_space)

    # ------------------

    results, _ = self.tuning_loop(hyperparameters_tuning, laundering_values_vali)

    return results







# place this inside the GNNMxiingManager? Probably. And maybe not actually
def get_gnn(manager, m_param):

    #self.m_param = self.model_configs.get('params')
    #self.m_settings = utils.get_tuning_configs(self.args).get('model_settings')
    #self.batch_size = self.m_param.get('batch_size')

    m_settings = fl_utils.get_tuning_configs(manager.args).get('model_settings')
    #m_param = hyperparams
    
    #n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    #e_dim = (sample_batch.edge_attr.shape[1] - e_dim_adjust) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - e_dim_adjust)
    #e_dim = (sample_batch.edge_attr.shape[1]) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    e_dim_adjust = 1 if m_settings.get('include_time') else 2
    node_features = manager.parties[manager._smallest_bank].data['train_data']['df'].x.shape[1] # switch between tuning / training?
    e_dim = (manager._num_features - e_dim_adjust)

    #manager.args['fl_parser'].model

    #using a registery
    model_name = manager.args['fl_parser'].model
    #if model_name not in self.MODEL_REGISTRY:
    #    raise ValueError(f"Unknown algo type: {model_name}")
    #return self.MODEL_REGISTRY[model_name]
    test123 = TEST_GNN_REGISTY[model_name]
    TEST_REGISTY[model_name]



    arguments = {'num_features': node_features, 'num_gnn_layers': m_param['params'].get('gnn_layers'),
                 'n_classes': 2, 'n_hidden': m_param['params'].get('hidden_embedding_size'),
                 'residual': False, 'edge_updates': manager.args['gnn_parser'].emlps, 
                 'edge_dim': e_dim, 'dropout': m_param['params'].get('dropout'), 
                 'final_dropout': m_param['params'].get('dropout')}
    
    test123(**arguments)


    gnn = gnn_m.GINe(
                num_features=node_features,
                num_gnn_layers=m_param['params'].get('gnn_layers'), 
                n_classes=2,
                n_hidden=m_param['params'].get('hidden_embedding_size'), 
                residual=False, edge_updates=manager.args['gnn_parser'].emlps, 
                edge_dim=e_dim,
                dropout=m_param['params'].get('dropout'), 
                final_dropout=m_param['params'].get('dropout'))
    
    return gnn



# tuning for regular
def tuning(manager, laundering_values_vali):

    # probably need to first init hyparameters here
    hyperparameters_tuning = manager.init_hyperparams()

    # --------------------------------
    for bank_id, party in manager.parties.items():
        party.prep_data_tuning()
    
    results, _ = manager.tuning_loop(hyperparameters_tuning, laundering_values_vali)

    return results


def tuning(manager, laundering_values_vali):


    # --------------------------------
    
    # for boost it would want it inside the hyperparameter loop.
    # however, if spliting and processing several datapoints at once, then
    # maybe it could be avoided, however too much memory usage?
    # So don't do that, find solution to only do once for reg and gnn?
    # will see at booster trees

    # preferable this should be a 'reuseable loop' for all the models
    for bank_id, party in manager.parties.items():
        party.prep_data_tuning()

    # probably need to first init hyparameters here

    # need to do this several times for -gnn
    hyperparameters_tuning = manager.init_hyperparams()


    # this part here is only needed for gnn --------------------------

    _, scores = tuning_loop(manager, hyperparameters_tuning, laundering_values_vali)
    params_to_keep = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    top_parameters = [hyperparameters_tuning[i] for i in params_to_keep]
    sample_space = _get_search_space(top_parameters)

    init_hp = manager.init_hyperparams(sample_space)
    hyperparameters_tuning = init_hp['hp_list']

    # ------------------

    results, _ = tuning_loop(manager, hyperparameters_tuning, laundering_values_vali)

    return results

# copy
def tuning_loop(manager, hyperparameters_tuning, laundering_values_vali):

    best_f1 = -1
    best_hyperparameters = None
    best_w = None
    best_metrics = None
    best_preditcions = None
    scores = []

    hyperparams = hyperparameters_tuning[0]
    # preferable this should be a 'reuseable loop' for all the models
    for hyperparams in hyperparameters_tuning:
        
        # initiate models:
        manager.init_models(hyperparams)

        # set initi parameters
        manager.get_global_w()
        #manager.global_w = init_w

        # training set initial parameters and get relevant data
        manager.send_global_w_params()

        # if reg or graph epochs is used. Or also is for decision trees, yes?
        # just update in one, and then another for sending to manager?
        #laundering_values = laundering_values_vali
        results = hf.fl_training(manager, laundering_values_vali)
        
        if results['metrics']['f1'] > best_f1:
            best_hyperparameters = hyperparams
            best_w = results['w']
            best_metrics = results['metrics']
            best_preditcions = copy.deepcopy(results['laundering_values'])
            best_f1 = results['metrics']['f1']
        
        scores.append(results['metrics']['f1'])

    return [{'best_hyperparameters': best_hyperparameters, 
            'best_w': best_w, 'best_metrics': best_metrics,
            'best_preditcions': best_preditcions}, scores]
    




def tuning(manager, laundering_values_vali):

    # probably need to first init hyparameters here
    init_hp = manager.init_hyperparams()
    hyperparameters_tuning = init_hp['hp_list']


    # --------------------------------
    best_f1 = -1
    best_hyperparameters = None
    best_w = None
    best_metrics = None
    best_preditcions = None

    
    # for boost it would want it inside the hyperparameter loop.
    # however, if spliting and processing several datapoints at once, then
    # maybe it could be avoided, however too much memory usage?
    # So don't do that, find solution to only do once for reg and gnn?
    # will see at booster trees

    # preferable this should be a 'reuseable loop' for all the models
    for bank_id, party in manager.parties.items():
        party.prep_data_tuning()
    
    # preferable this should be a 'reuseable loop' for all the models
    for hyperparams in hyperparameters_tuning:
        
        # initiate models:
        manager.init_models(hyperparams)

        # set initi parameters
        manager.get_global_w()
        #manager.global_w = init_w

        # training set initial parameters and get relevant data
        manager.send_global_w_params()

        # if reg or graph epochs is used. Or also is for decision trees, yes?
        # just update in one, and then another for sending to manager?
        results = hf.fl_training(manager, laundering_values_vali)
        
        if results['metrics']['f1'] > best_f1:
            best_hyperparameters = hyperparams
            best_w = results['w']
            best_metrics = results['metrics']
            best_preditcions = copy.deepcopy(results['laundering_values'])
            best_f1 = results['metrics']['f1']

    return {'best_hyperparameters': best_hyperparameters, 
            'best_w': best_w, 'best_metrics': best_metrics,
            'best_preditcions': best_preditcions}
















def intiate_parameters(self, fl_args):

    # first get then number of features, this are kept or maybe they aren't?

    if fl_args.model_type == 'graph':

        # this works, but potentially I should change such that it request from a party how many
        # features are in the end, however, this can wait till more actual fl is implemented

        x_0 = fl_utils.get_tuning_configs(parsers)['individual_banks']['small']['x_0']
        params_list = [fl_utils.hyper_sampler(parsers['fl_parser'], 5000, None) for i in range(x_0)]

        #
        init_w = None

    return {'init_w': init_w, 'hp_list': hp_list}



# should parties models be 'removed/reset' before setting new one, in order to ensure old
# one is removed?

def init_models(self):
                
    if fl_args.model_type == 'graph':
        
        # can't share batchnorm parameters? Use layernorms or groupnorm instead, use differential privacy

        # need to save the parameters, need to first do 'small' processing of a dataset
        # to get the amount of features.

        # how to handl num_features?
        models = {bank_id: gnn_m.GINe(
            num_features=10, num_gnn_layers=tmp_params['params'].get('gnn_layers'), n_classes=2,
            n_hidden=tmp_params['params'].get('hidden_embedding_size'), residual=False, edge_updates=gnn_args.emlps, edge_dim=2,
            dropout=tmp_params['params'].get('dropout'), final_dropout=tmp_params['params'].get('dropout')
        ) for bank_id in manager.parties.keys()}


        # model 0's values are used as initiation values

        params_update = []
        init_parameters = {}
        bank_0 = next(iter(manager.parties.keys()))

        # use learnable only
        for name, param in models[bank_0].named_parameters():
            #init_parameters[name] = param
            params_update.append(name)

        
        # use learnable and non-learnable
        for name, param in models[bank_0].state_dict().items():
            #init_parameters[name] = param
            params_update.append(name)


        for name in params_update:
            for bank_id, model in models.items():
                if bank_id == bank_0: continue
                if model.state_dict()[name].shape[0] > 0:
                    for j in range(0, model.state_dict()[name].shape[0]):
                        model.state_dict()[name][j] = models[bank_0].state_dict()[name][j]
                else:
                    model.state_dict()[name] = models[bank_0].state_dict()[name]

        # need the manager to set the models for the parties

        # sender models / parameters
        for bank_id, party in manager.parties.items():
            party.model = models[bank_id]

    return 0













from abc import ABC, abstractmethod
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




# setup
utils.logger_setup()
parser = utils.get_parser()
args = parser.parse_args()

utils.set_seed(args.seed, True)
args.scenario = 'individual_banks'
fr_banks, sr_banks = get_relevant_banks(args)
# tmp stuff
args.model = 'xgboost'
model = Model.from_model_type(args)


# prepepping of data
df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{args.size}' + f'_{args.ir}' + '.csv')
raw_data = get_data(df, model.args, split_perc = split_perc)


fr_banks = fr_banks[0:5]
dic_train_data, dic_vali_data, dic_test_data = {}, {}, {}
for index, bank in enumerate(fr_banks):

    model = Model.from_model_type(args)

    bank_indices = model.get_indices(raw_data, bank=bank)
    train_data, vali_data, test_data = model.get_data(raw_data, bank_indices)
    dic_train_data[bank], dic_vali_data[bank], dic_test_data[bank] = train_data, vali_data, test_data


# next feature engineering is needed
# if methods includes not sharing even encrypted data, this needs to be done. Which is most methods
# but have this in the party class?


class BaseFL:

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def receive_handle_messages(self):
        pass

    @abstractmethod
    def send_messages(self):
        pass


class Manager(BaseFL):
    model_REGISTRY = None
    fl_REGISTRY = None
    
    def __init__(self, args):
        super().__init__(args)
        self.model = 0
        self.fl_method = 0
        self.parties = []

    
    def _create_model(self, args):
        model_type = utils.model_types.get(args.model)
        if model_type not in self.model_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.model_REGISTRY[model_type]


    def _create_fl_method(self, args):
        fl_method = utils.fl_methods.get(args.fl_method)
        if fl_method not in self.fl_REGISTRY:
            raise ValueError(f"Unknown FL method: {fl_method}")
        return self.fl_REGISTRY[fl_method]

    # 
    def send_messages(self):  
        return super().send_messages()
    
    def initiation(cls, args):
        # here a function that 'creates' the base model and starting parameters is needed.
        # this model then needs to be sent to the party class(es)
        test_model = 123
        cls.send_messages(test_model)
        return 0


class Party(BaseFL):

    def __init__(self):
        pass

    def send_messages(self):
        return super().send_messages()
    
    def calculations(self):
        self.send_messages()

global_dict = {}

parties = {}
for index, bank in enumerate(fr_banks):

    model = Model.from_model_type(args)
    bank_indices = model.get_indices(raw_data, bank=bank)
    train_data, vali_data, test_data = model.get_data(raw_data, bank_indices)
    dic_train_data[bank], dic_vali_data[bank], dic_test_data[bank] = train_data, vali_data, test_data

    # need to at the data at it
    parties[bank] = Party()







    





"""


    # under construction, need to make several changes here
    def get_model(self, args):
        self.model_type = utils.model_types.get(args.model)
        if self.model_type not in self.model_REGISTRY:
            raise ValueError(f"Unknown model type: {self.model_type}")
        base_model = self.model_REGISTRY[self.model_type]
        self.model = base_model.from_model_type(args)
    
    # under construction, need to make several changes here
    def get_flmethod_model(self, args):
        self.FL_method = utils.FL_method.get(args.model)
        if self.FL_method not in self.fl__REGISTRY:
            raise ValueError(f"Unknown model type: {self.FL_method}")
        base_method = self.fl__REGISTRY[self.FL_method]
        self.flmethod = base_method.from_model_type(args)
        


"""







