# configs.py

include_time = False
split_perc=[0.60, 0.20]
epochs = 100
#epochs_fl = 30

# Lazy evaluation to avoid circular import
_save_direc_training = None

def __getattr__(name):
    """Module-level __getattr__ for lazy evaluation of save_direc_training."""
    if name == 'save_direc_training':
        global _save_direc_training
        if _save_direc_training is None:
            import utils
            _save_direc_training = utils.get_data_path() + '/AML_work_study/experiments/'
        return _save_direc_training
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
