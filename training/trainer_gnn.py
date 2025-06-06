import copy
import tqdm
import torch
import training.gnn_utils as tgu
import configs.configs as config
from sklearn.metrics import f1_score
import warnings
import utils
from data.feature_engi import general_feature_engineering
from abc import ABC, abstractmethod


def train_gnn(args, train_data, test_data, hyperparameters, seeds = 4):

    models = {}
    for seed in range(1, seeds + 1):
        trainer = GNNTrain(args, train_data, test_data, hyperparameters)
        utils.set_seed(seed)
        model, f1 = trainer.train()
        models[f'seed_{seed}'] = None
        models[f'seed_{seed}'] = {'model': model, 'f1': f1}

    return models

class GNNTrain(ABC):
    def __init__(self, args, train_data, test_data, model_configs, seed = None):
        
        self.args = args
        self.seed = seed
        self.train_indices = train_data['pred_indices']
        self.pred_indices = test_data['pred_indices']
        self.train_data = train_data['df']
        self.test_data = test_data['df']
        self.epochs = 2 if args.testing else config.epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_configs = model_configs
        
        self._init_model_config()
        self.get_model_loss()

    def _init_model_config(self):
        self.m_param = self.model_configs.get('params')
        self.m_settings = utils.get_tuning_configs(self.args).get('model_settings')
        self.batch_size = self.m_param.get('batch_size')

    def get_model_loss(self):
        self.model = tgu.get_model(self.train_data, self.m_param, self.m_settings, self.args)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.m_param.get('learning rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([self.m_param.get('w_ce1'), self.m_param.get('w_ce2')]).to(self.device))

    def train(self):

        if self.seed:
            utils.set_seed(self.seed)
        
        if self.batch_size > 1:
            return self._train_with_batches()
        
        else:
            return self._train_no_batching()

    def _train_with_batches(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_loader, test_loader = tgu.get_loaders(self.train_data, self.test_data, self.pred_indices, self.m_param, self.batch_size)

        best_val_f1 = -1
        best_model_state = None

        for epoch in range(self.epochs):

            self.model.train()

            for batch in tqdm.tqdm(train_loader, disable=not self.args.tqdm):
                
                self.optimizer.zero_grad()
                target_edge_attr = self.train_data.edge_attr[batch.input_id, :].to(self.device)
                batch.edge_attr = batch.edge_attr[:, 1:] if self.m_settings['include_time'] else batch.edge_attr[:, 2:]
                target_edge_attr = target_edge_attr[:, 1:] if self.m_settings['include_time'] else target_edge_attr[:, 2:]
                batch.to(self.device)

                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, target_edge_attr)
                pred = out
                ground_truth = batch.edge_label

                loss = self.loss_fn(pred, ground_truth)
                loss.backward()
                self.optimizer.step()

            
            current_f1 = GNNEval.eval_batched(self, test_loader)
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_model_state = copy.deepcopy(self.model.state_dict())

        return best_model_state, best_val_f1


    def _train_no_batching(self):

        train_data = copy.deepcopy(self.train_data)
        train_data.edge_attr = train_data.edge_attr[:, 1:] if self.m_settings['include_time'] else train_data.edge_attr[:, 2:]
        train_data.to(self.device)

        best_val_f1 = -1
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()

            self.optimizer.zero_grad()
            pred = self.model(train_data.x, train_data.edge_index, train_data.edge_attr, train_data.edge_index, train_data.edge_attr)
            loss = self.loss_fn(pred, train_data.y)

            loss.backward()
            self.optimizer.step()

            current_f1 = GNNEval.eval_no_batching(self)
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_model_state = copy.deepcopy(self.model.state_dict())

        return best_model_state, best_val_f1


class GNNEval:

    @staticmethod
    def eval_batched(trainer, loader):

        #trainer = self
        #loader = test_loader

        model = trainer.model
        data = trainer.test_data
        inds = trainer.pred_indices
        args = trainer.args
        device = trainer.device
        m_settings = trainer.m_settings

        preds = []
        ground_truths = []
        model.eval()

        for batch in tqdm.tqdm(loader, disable=not args.tqdm):
            
            #select the seed edges from which the batch was created
            inds = inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
 
            # as in the training function
            target_edge_attr = data.edge_attr[batch_edge_inds, :].to(device)
            batch.edge_attr = batch.edge_attr[:, 1:] if m_settings['include_time'] else batch.edge_attr[:, 2:]
            target_edge_attr = target_edge_attr[:, 1:] if m_settings['include_time'] else target_edge_attr[:, 2:]
            
            with torch.no_grad():
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, target_edge_attr)
                #out = out
                ground_truth = batch.edge_label
                    
                pred = out.argmax(dim=-1)
                preds.append(pred)
                ground_truths.append(ground_truth)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        f1 = f1_score(ground_truth, pred, average='binary', zero_division=0)

        return f1
    
    @staticmethod
    def eval_no_batching(trainer):
        
        #trainer = self
        model = trainer.model
        data = copy.deepcopy(trainer.test_data)
        inds = trainer.pred_indices
        device = trainer.device
        m_settings = trainer.m_settings

        data.edge_attr = data.edge_attr[:, 1:] if m_settings['include_time'] else data.edge_attr[:, 2:]

        model.eval()
        with torch.no_grad():
            data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.edge_index, data.edge_attr)
            out = out[inds]
            pred = out.argmax(dim=-1)
        
        pred = pred.cpu().numpy()
        ground_truth = data.y[inds].cpu().numpy()
        f1 = f1_score(ground_truth, pred, average='binary', zero_division=0)

        return f1
        



        




















