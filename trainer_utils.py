import json
import trainers as trs
import process_data_type as pdt

def get_model_configs(args):
    with open('model_configs.json', 'r') as file:
        model_parameters = json.load(file)
    return model_parameters.get(args.model)

#{"params": {"xgb_parameters": {"objective":  "binary:logistic", "eval_metric": "logloss", "max_depth": 6, "learning_rate": 0.01 }, "num_rounds": 100} }

trainer_classes = {
    'graph': trs.train_gnn_trainer,
    "booster": trs.xgboost_trainer,
    #"simple_nn_full": simple_nn_trainer
}

model_types = {
    'GINe': 'graph',
    'xgboost': 'booster'
}

file_types = {
    'graph': 'pth',
    'booster': 'pkl'
}

data_functions = {
    'graph': pdt.process_graph_data,
    'booster': pdt.process_regular_data,
    #'graph_data': pdt.process_graph_data,
    #'regular_data': pdt.process_regular_data,
}

data_types = {
    'graph': 'graph_data',
    'booster': 'regular_data'
}

