

FL_ALGO_REGISTRY_PARTY = {}
def regi_algo_party(name):
    def wrapper(cls):
        FL_ALGO_REGISTRY_PARTY[name] = cls
        return cls
    return wrapper


FL_ALGO_REGISTRY_MANAGER = {}
def regi_algo_manager(name):
    def wrapper(cls):
        FL_ALGO_REGISTRY_MANAGER[name] = cls
        return cls
    return wrapper

FL_REG_MODEL_REGISTRY = {}
def register_reg_model(name):
    def wrapper(cls):
        if name in FL_REG_MODEL_REGISTRY:
            raise ValueError(f"Model {name} already registered!")
        FL_REG_MODEL_REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return wrapper


GNN_REGISTRY = {}
def register_gnn(name):
    def wrapper(cls):
        if name in GNN_REGISTRY:
            raise ValueError(f"Model {name} already registered!")
        GNN_REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return wrapper



# currently not used, save for later
"""FL_MODEL_TYPE_REGISTRY_MANAGER = {}
def regi_model_type_manager(name):
    def wrapper(cls):
        FL_MODEL_TYPE_REGISTRY_MANAGER[name] = cls
        return cls
    return wrapper"""




