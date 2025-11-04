
import configs.configs as configs
import copy
import federated_learning.inference as flin
from federated_learning.inference import metrics

def fl_training(manager, laundering_values):

    # maybe include hyperparameters here, like as input to the function here

    best_w = None
    best_metrics = None
    best_preditcions = None
    best_f1 = -1

    epochs = 20 if manager.args['data_parser'].testing else configs.epochs_fl

    for i in range(0, epochs):

        for bank_id, party in manager.parties.items():    
            # little unsure if I should update parameters like this, or if 
            # it should be kept inside the party

            party.update_local_w()
            party.send_local_w(manager)
        
        # manager update global_w
        manager.update_global_w()

        # send global_w
        manager.send_global_w()

        # I need to reset laundering_values or something every time new parameters are tested
        # inference / status
        if (i+1) % 20 == 0:

            # reset preditcions
            laundering_values['predictions_fl'] = 0
            
            for bank_id, party in manager.parties.items():
                flin.update_laundering_values(party, laundering_values)

            tmp_metrics = metrics(laundering_values['true_y'], laundering_values['predictions_fl'])

            if tmp_metrics['f1'] > best_f1:
                best_w = manager.global_w
                best_metrics = tmp_metrics
                best_preditcions = copy.copy(laundering_values['predictions_fl'])
                best_f1 = tmp_metrics['f1']
                
    laundering_values['predictions_fl'] = best_preditcions

    return {'w': best_w, 'metrics': best_metrics, 'laundering_values': laundering_values}



