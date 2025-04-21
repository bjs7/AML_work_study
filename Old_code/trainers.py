from sklearn.metrics import log_loss
from abc import ABC, abstractmethod
import json
import xgboost as xgb
import trainer_utils as tu
import trainer_gnn as tgnn
import torch

class xgboost:
    def __init__(self, args, model_configs):
        self.args = args
        self.model_configs = model_configs
    
    def train(self, train_data):
        dtrain = xgb.DMatrix(train_data['X'], train_data['y'])

        self.model = xgb.train(self.model_configs['params'], dtrain, 
                               self.model_configs['num_rounds'])

    







# old -------------------------------------------------------------------------------------------------------------

class BaseTrainer(ABC):
    def __init__(self, args, data, model_configs): #, labels
        self.args = args
        self.data = data
        self.model_configs = model_configs
        #self.labels = labels

    @abstractmethod
    def train(self):
        pass

class train_gnn_trainer(BaseTrainer):

    def __init__(self, args, data, model_configs):
        self.args = args
        self.data = data
        self.model_configs = model_configs

    def train(self):
        self.model = tgnn.train_gnn(self.args, self.data, self.model_configs)
        #return self.model
        #return model_wrap(self.model, {"bank_indices": self.data.get("bank_indices", None), "scaler": self.data.get("scaler", None), "encoder_pay": self.data.get("encoder_pay", None), "encoder_cur": self.data.get("encoder_cur", None)})
        return model_wrap(self.model, {"pred_indices": self.data.get("pred_indices", None), 'scaler_encoders': self.data.get('scaler_encoders', None)})

class model_wrap:

    def __init__(self, model, data):
        self.model = model
        self.bank_indices = data.get('bank_indices')
        self.scaler_encoders = data.get('scaler_encoders')

        #self.scaler = data.get('scaler', None)
        #self.encoder_pay = data.get('encoder_pay', None)
        #self.encoder_cur = data.get('encoder_cur', None)

    def predict(self, X):
        predict_function = predict_functions.get(self.model.__name__)
        return predict_function(self.model, X)


class xgboost_trainer(BaseTrainer):

    def __init__(self, args, data, model_configs):
        super().__init__(args, data, model_configs)

        #model_configs = tu.get_model_configs(self.args)
        self.params = self.model_configs['params']
        self.num_rounds = self.model_configs['num_rounds']

    def train(self):
        dtrain = xgb.DMatrix(self.data['X'], self.data['y'])
        self.model = xgb.train(self.params, dtrain, self.num_rounds)

        return model_wrap(self.model, {"bank_indices": self.data.get("bank_indices"), "scaler": self.data.get("scaler", None), "encoder_pay": self.data.get("encoder_pay", None), "encoder_cur": self.data.get("encoder_cur", None)})



class simple_nn_trainer(BaseTrainer):

    def __init__(self, model, data, epochs = 100, lr=0.01, momentum=0.9):
        super().__init__(model, data)
        self.epochs = epochs
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def train(self):
        self.model.train()
        prev_loss = float('inf')
        epsilon = 0.0001
        no_improvement = 0

        X_tensor = torch.tensor(self.data.values, dtype = torch.float32)
        y_tensor = torch.tensor(self.labels.values, dtype = torch.float32).reshape(-1, 1)

        for epoch in range(self.epochs):

            self.optimizer.zero_grad()
            predictions = self.model(X_tensor)

            loss = self.criterion(predictions, y_tensor)
            loss.backward()
            self.optimizer.step()

            loss_values = loss.item()

            print(f"Epoch {epoch + 1} - Losses: {loss_values}")

            if abs(prev_loss - loss_values) < epsilon:
                no_improvement += 1
                if no_improvement == 4:
                    break
            else:
                no_improvement = 0

            prev_loss = loss_values

        return self.model #model_wrap(self.model)
