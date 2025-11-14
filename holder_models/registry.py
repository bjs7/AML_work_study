

# Class/model type register --------------------------------------------------------------------------------------------------------------

# Standard
MODEL_TYPE_REGISTRY = {}
def regi_model_types(name):
    def wrapper(cls):
        MODEL_TYPE_REGISTRY[name] = cls
        return cls
    return wrapper

# Inference
INFERENCE_REGISTRY = {}
def regi_infer_types(name):
    def wrapper(cls):
        INFERENCE_REGISTRY[name] = cls
        return cls
    return wrapper


# Booster register -----------------------------------------------------------------------------------------------------------------------

BOOSTER_REGISTRY = {}
def regi_booster(name):
    def wrapper(cls):
        BOOSTER_REGISTRY[name] = cls
        return cls
    return wrapper








