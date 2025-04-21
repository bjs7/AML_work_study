
class BaseLoader(ABC):
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def load(self):
        pass




# load models

def list_trained_types(model):
    directory_path = os.path.join(configs.save_direc_training, model)
    folders = [name for name in os.listdir(directory_path)  if os.path.isdir(os.path.join(directory_path, name))]
    return folders

def load_model(model, model_folder):
    directory_path = os.path.join(configs.save_direc_training, model + '/' + model_folder)
    return directory_path
    model_type = tu.model_types.get('GINe')

    if model_type == 'graph':
        #model = 0
        return 0
    
    
    #get_model


    return directory_path

def get_model_configs(folder):
    with open(folder + '/configurations.json', 'r') as file:
        model_configs = json.load(file)
    return model_configs

test = 'GINe'
list_configs = list_trained_types('GINe')
print(list_configs)
model_folder = list_configs[0]
test123 = load_model('GINe', model_folder)
model_configs = get_model_configs(test123)

#bank_models = [name for name in os.listdir(test123) if re.findall(r"bank_\d+", name)]
#bank_models = re.findall(r"bank_\d+", test1)
models_to_load = [name for name in os.listdir(test123) if os.path.splitext(name)[1] == '.pth']

# foerst faa model og saa load data for hver bank/model
# nvm kan ikke. skal bruge infer_gnn for hver model

# type
model_type = tu.model_types.get('GINe')

# load vali/test data that is used for all / overall
data_processor = tu.data_functions.get(model_type)
data_for_indices = pd.concat([data['regular_data']['train_data']['x'][['From Bank', 'To Bank']], data['regular_data']['vali_data']['x'][['From Bank', 'To Bank']]], axis=0) if model_type == 'graph' else data['regular_data']['vali_data']['x'][['From Bank', 'To Bank']]
data = data[tu.data_types.get(model_type)]['vali_data']


# now load each model
vali_data = data[tu.data_types.get(model_type)]['vali_data']











#data = data[tu.data_types.get(model_type)]['train_data']

data_for_indices = pdt.get_bank_indices(data_for_indices, bank) if bank else data_for_indices.index.tolist()
train_data = data_processor(data, data_for_indices, args)



def infer_gnn(model_configs, vali_data, **kwargs):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###################################################################
    ##### Might be come relevant to make some changes here later! #####
    ###################################################################

    m_param = model_configs.get('model_configs').get('params')
    m_settings = model_configs.get('model_configs').get('model_settings')

    # need to make dynamic here?
    num_neighbors = m_param.get('num_neighbors')
    batch_size = m_param.get('batch_size')[0] if vali_data.num_nodes < 10000 else m_param.get('batch_size')[1]
    nn_size = len(num_neighbors)

    # loader
    #transform = partial(account_for_time, main_data=train_data)
    vali_loader = LinkNeighborLoader(vali_data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False, transform=None)
    sample_batch = next(iter(vali_loader))
    model = tgu.get_model(sample_batch, nn_size)

    

    # setup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=m_param.get('lr'))
    sample_batch.to(device)



    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([m_param.get('w_ce1'), m_param.get('w_ce2')]).to(device))
    #model = train_homo(vali_loader, train_data, train_indices, model, optimizer, loss_fn, device, m_settings)

    #model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, **kwargs)
    #model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device)

    return model
