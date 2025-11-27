#2. Hyperparameter Reuse System
#For the hyperparameter reuse, I suggest two approaches: Approach A: Load from previous experiment Add this to your manager's tuning method:
def tuning(self, laundering_values, use_hyperparameters_from=None):
    """
    Args:
        laundering_values: Data for tuning
        use_hyperparameters_from: Optional path to previous experiment to reuse hyperparameters
    """
    results = {}
    self.set_mode('tuning')
    
    # Option to load hyperparameters from previous experiment
    if use_hyperparameters_from is not None:
        loaded_results = load_experiment(use_hyperparameters_from)
        
        # For full_info, just return the loaded hyperparameters
        if self.args['fl_parser'].fl_algo == 'full_info':
            return {'hyperparameters': loaded_results.hyperparameters, 'f1_score': 1}
        
        # For individual, return hyperparameters for each bank
        # (assumes same banks, could add validation)
        for bank_id in self.parties.keys():
            results[bank_id] = {
                'hyperparameters': loaded_results.hyperparameters.get(bank_id, loaded_results.hyperparameters),
                'f1_score': 1
            }
        return results
    
    # Regular tuning logic
    for bank_id, party in self.parties.items():
        if self.args['data_parser'].ibm_hp:
            tuned_hyparameters, f1_score_for_hp = ibm_gnn, 1
        else:
            party.prep_data()
            party_laundering_values = self._helper_party_tuning(party, laundering_values)
            tuned_hyparameters, f1_score_for_hp = self._gnn_tuning(party_laundering_values, bank_id=bank_id)

        results[bank_id] = {'hyperparameters': tuned_hyparameters, 'f1_score': f1_score_for_hp}
    
    return results


if self.args['data_parser'].ibm_hp:
    # If ibm_hp is a string path, load from that experiment
    if isinstance(self.args['data_parser'].ibm_hp, str):
        from relbanks_saving_analysis.load_results import load_experiment
        loaded = load_experiment(self.args['data_parser'].ibm_hp)
        tuned_hyparameters = loaded.hyperparameters.get(bank_id, loaded.hyperparameters)
    else:
        # If ibm_hp is True, use predefined ibm_gnn
        tuned_hyparameters = ibm_gnn
    f1_score_for_hp = 1



#Approach B: Hyperparameter registry (more sophisticated) Create a separate file hyperparameter_registry.json:
{
  "GINe": {
    "small_HI": {
      "default": {...},
      "ibm_fe": {...}
    },
    "medium_LI": {
      "default": {...}
    }
  }
}
#Then add helper functions:
# In save_results.py or utils.py
def save_to_hyperparameter_registry(manager, hyperparameters, registry_path='hyperparameter_registry.json'):
    """Save hyperparameters to global registry for reuse."""
    # Create key from configuration
    model = manager.args['fl_parser'].model
    data_key = f"{manager.args['data_parser'].size}_{manager.args['data_parser'].ir}"
    
    data_flags = ['ibm_fe', 'ibm_hp', 'train_for_final']
    data_settings = [flag for flag in data_flags if getattr(manager.args['data_parser'], flag)]
    settings_key = '__'.join(data_settings) if data_settings else 'default'
    
    # Load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
    
    # Update registry
    if model not in registry:
        registry[model] = {}
    if data_key not in registry[model]:
        registry[model][data_key] = {}
    
    registry[model][data_key][settings_key] = hyperparameters
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)

def load_from_hyperparameter_registry(manager, registry_path='hyperparameter_registry.json'):
    """Load hyperparameters from global registry."""
    if not os.path.exists(registry_path):
        return None
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    model = manager.args['fl_parser'].model
    data_key = f"{manager.args['data_parser'].size}_{manager.args['data_parser'].ir}"
    
    data_flags = ['ibm_fe', 'ibm_hp', 'train_for_final']
    data_settings = [flag for flag in data_flags if getattr(manager.args['data_parser'], flag)]
    settings_key = '__'.join(data_settings) if data_settings else 'default'
    
    try:
        return registry[model][data_key][settings_key]
    except KeyError:
        return None