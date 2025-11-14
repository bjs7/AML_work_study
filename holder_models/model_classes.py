import xgboost as xgb
import lightgbm as lgb


# Boosters
class XGBoostMixin:
    def pred_data(self, train_data):
        return xgb.DMatrix(train_data['x'], train_data['y'])

    def train_model(self, params, train_data, num_rounds):
        return xgb.train(params, train_data, num_rounds)

    def predict(self, model, vali_data):
        return model.predict(xgb.DMatrix(vali_data['x']))
    
    

class LightGBMMixin:
    def pred_data(self, train_data):
        return lgb.Dataset(train_data['x'], train_data['y'])
    
    def train_model(self, params, train_data, num_rounds):
        return lgb.train(params, train_data, num_boost_round=num_rounds)
    
    def predict(self, model, vali_data):
        return model.predict(vali_data['x'])