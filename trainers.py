from sklearn.metrics import log_loss
from abc import ABC, abstractmethod
import json
import xgboost as xgb
import trainer_utils as tu
import trainer_gnn as tgnn
import torch


class BaseTrainer(ABC):
    def __init__(self, args, data): #, labels
        self.args = args
        self.data = data
        #self.labels = labels

    @abstractmethod
    def train(self):
        pass

class train_gnn_trainer(BaseTrainer):

    def __init__(self, args, data):
        self.args = args
        self.data = data

    def train(self):
        self.model = tgnn.train_gnn(self.args, self.data)
        return self.model

class model_wrap:

    def __init__(self, model, data):
        self.model = model
        self.bank_indices = data.get('bank_indices')
        self.scaler = data.get('scaler', None)

    def predict(self, X):
        predict_function = predict_functions.get(self.model.__name__)
        return predict_function(self.model, X)


class xgboost_trainer(BaseTrainer):

    def __init__(self, args, data):
        super().__init__(args, data)
        
        self.args = args
        self.data = data

        model_configs = tu.get_model_configs(self.args)
        self.params = model_configs['params']['xgb_parameters']
        self.num_rounds = model_configs['params']['num_rounds']
        #self.X = data['X']
        #self.y = data['y']

        #self.params = params
        #self.num_rounds = num_rounds

    def train(self):
        dtrain = xgb.DMatrix(self.data['X'], self.data['y'])
        self.model = xgb.train(self.params, dtrain, self.num_rounds)

        return model_wrap(self.model, {"bank_indices": self.data.get("bank_indices"), "scaler": self.data.get("scaler", None)})


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
