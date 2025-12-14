




        


# this function would not really be relevant for a real setting, banks should be able to "track"
# or extract the overlaps they have with other or certain banks. There probably isn't a "global"
# overview of all the transaction that has happened and then the banks can see which one they
# participated in. So banks should be able to say "these transactions happended with this bank"
# so this/these functions wouldn't be relevant in a real setting


class FLVertical:

    def __init__(self, manager, model):
        self.manager = manager
        self.model = model
        

    def set_up(self):  
        for bank_id, party in self.manager.parties.items():
            #party.prep_data()
            party._get_shared_edge_indices()

        for bank_id, party in self.manager.parties.items():
            party.get_ints_send_ints()

        for bank_id, party in self.manager.parties.items():
            party.get_node_mappings()

        for bank_id, party in self.manager.parties.items():
            party.received_embeddings = {}

    def parties_run_gnn_layer(self, layer_idx):

        for bank_id, party in self.manager.parties.items():

            if layer_idx == 0:
                party.current_embeddings = self.model.emed_features(party.data['train_data']['df'].x,
                                                                    party.data['train_data']['df'].edge_attr)
            

            party.current_embeddings = self.model.apply_gnn_layer(party.current_embeddings['nodes'],
                                                            party.current_embeddings['edges'],
                                                            party.data['train_data']['df'].edge_index,
                                                            layer_idx)
            
    def merg_embeddings(self):

        for bank_id, party in self.manager.parties.items():
            party.send_embeddings()

        for bank_id, party in self.manager.parties.items():
            party.merge_embeddings()

    def full_model(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams.get('learning_rate'))
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([hyperparams.get('w_ce1'), hyperparams.get('w_ce2')]).to(device)) 

        self.optimizer.zero_grad()
        
        for i in range(self.model.num_gnn_layers):
            self.parties_run_gnn_layer(i)
            self.merg_embeddings()
        
        self.manager.embeddings_indices = {}
        for bank_id, party in self.manager.parties.items():
            self.manager.embeddings_indices[bank_id] = {
                'embeddings': self.model.prep_nodes_edges(party.current_embeddings['nodes'],
                                                            party.current_embeddings['edges'],
                                                            party.data['train_data']['df'].edge_index),
                'indices': party.data['train_data']['indices']
            }
        
        predictions = []
        for idx in self.manager.data['train_data']['indices']:
            
            combined_embeds = combine_embeds_from_banks_with_idx(idx)
            prediction = self.model.final_layer(combined_embeds)
            predictions.append(prediction)

        loss = loss_fn(true_y, predictions)
        loss.backward()
        self.optimizer.step()





 

    

num_edge_features = manager.parties[0].data['train_data']['df'].edge_attr.shape[1]
embeds = Embeddings(num_features=1, edge_dim=num_edge_features, num_gnn_layers=2)
manager.embedder = embeds


Flvert = FLVertical(manager, embeds)


Flvert.set_up()
Flvert.full_model()


# so all nodes and edges are fully update in the end after last gnn_layer.
# banks send their end of a transaction, like after reforming them, so (1,300) shaped tensor,
# take half and half of them, combine, or concate, but any, then that is used to make the prediction
# in the final mlp layer. But really need to think about and consider how to do this last step
# such that it is ensure that every node/edges is updated appropriately, like such that they get the
# info from all the k'th neighbours.

# Gradient needs to flow back and furth (need to read some more about this in FL literature),
# over the whole computation graph, that includes over all parties, input variables (everything) etc.
# such that pytorch or the backward/optimizer has the full computation graph. 
# 
# For the "first idea". Still let parties run their embeddings, send them back and furth.
# But it is not possible to only use final embeddings and just calculate gradient
# from the manager just running backpropagation on the network without input from the parties.
# So, instead parties send back and furth messages/embeddings, update the nodes, etc.
# and then pass, tensors in format of something like (1,300), to the manager, 
# who uses them for the final output/predictions. The manager will get a gradient
# for the last layer, send that back to the parties, and the parties then use this
# gradient for calculating the gradient of the remaining of the graph, as so
# the a "full" gradient at each party can be obtained, that they can then use to update the 
# parameters with, though the models are (most likely) not local, but its a full model
# that they all update the model parameters of, they though have to send that back to the
# manager? Though how do the manager "deal" with or handle the fact that
# once one party updates parameters, the gradients from another party may no longer be valid?
# That could be a reason for having local parts of the model? Need to check what has or is
# done in the literature. 



# Though, before implementing these things, see if they work
# with actually doing it without split gradients, like here on the computer where pytorch
# automatically have access to the computation graph and stuff can flow back and furth by itself
# without problems. So see if those designs or frameworks work, just in its pure form, 
# without accounting for real life settings. Just make sure that it can work in it purest form
# without introducing "realistic" constraints/settings, with free flow here on the computer.
# And test that it works here on the computer with where pytorch, gradient etc. can run through
# and has access to the computation graph and there is not split of learning, gradients, local 
# optimizers, local layers, sending gradients back and furth etc. just all on the computer

# when implementing this, make it "easy" or somewhat adjustable to be more "realistic",
# if it isn't too complicated


Flvert.manager.parties[0].current_embeddings['edges'].shape
Flvert.manager.parties[0].current_embeddings['nodes'][Flvert.manager.parties[0]._get_edge_indices()].shape





nodes = Flvert.manager.parties[0].current_embeddings['nodes'][Flvert.manager.parties[0].data['train_data']['df'].edge_index.T].reshape(-1, 2 * 100).relu()
out = torch.cat((nodes, Flvert.manager.parties[0].current_embeddings['edges'].view(-1, Flvert.manager.parties[0].current_embeddings['edges'].shape[1])), 1)

# ----------------------------------------------------------

# approach share embeddings with manager only, in more realistic setting gradients need to be
# sent from manager back to party


# when implementing this, make it "easy" or somewhat adjustable to be more "realistic",
# if it isn't too complicated
# so it should be something about the "send values maybe?"

# this approach might not be worth it, still a limited amount of jumps/neighbours that can be obtained?
# and what about the order of the gradient? Like when the parties get to "contribute" to it or
# update it. But that also needs to be considered in the other case?

class FLGNVertical:

    def __init__(self, manager, model):
        self.manager = manager
        self.model = model

    # could keep this seperate or like just here, and then still "load" the model
    # as in the other cases, like the parties get the model in that way
    # so still need to "setup" the parties as usual.
    # such that their data is ready, the right etc.
    # gets feature engineering, need to do these things before I move on, such that I have a
    # template etc.

    def get_embeddings(self):

        self.manager.embeddings_indices = {}
        for bank_id, party in self.manager.parties.items():

            party.current_embeddings = self.model.emed_features(party.data['train_data']['df'].x,
                                                                party.data['train_data']['df'].edge_attr)
            
            for layer_idx in range(self.model.num_gnn_layers):
                party.current_embeddings = self.model.apply_gnn_layer(party.current_embeddings['nodes'],
                                                                party.current_embeddings['edges'],
                                                                party.data['train_data']['df'].edge_index,
                                                                layer_idx)
        
            self.manager.embeddings_indices[bank_id] = {
                'embeddings': self.model.prep_nodes_edges(party.current_embeddings['nodes'],
                                                            party.current_embeddings['edges'],
                                                            party.data['train_data']['df'].edge_index)
            }


num_edge_features = manager.parties[0].data['train_data']['df'].edge_attr.shape[1]
embeds = Embeddings(num_features=1, edge_dim=num_edge_features, num_gnn_layers=2)
manager.embedder = embeds


Flvgnn = FLGNVertical(manager, embeds)

Flvgnn.get_embeddings()

Flvgnn.manager.embeddings_indices











class VerticalFLGNN(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                n_hidden=100, edge_updates=False, residual=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, batching=True):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden)), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden),))
            self.convs.append(conv)
            if batching:
                self.batch_norms.append(BatchNorm(n_hidden))
            else:
                self.batch_norms.append(LayerNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(25, n_classes))

    def forward_embeddings(self, nodes, edges):

        nodes = self.node_emb(nodes)
        edges = self.edge_emb(edges)

        return {'nodes': nodes, 'edges': edges}
    
    def forward_hidden(self, nodes, edges, edge_index):

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x

        return 0
    
    def forward_final(self, x):
        return self.mlp(x)
    
    def forward(self):

        return 0






#embeddings
class Embeddings(torch.nn.Module):

    def __init__(self, num_features = 1, n_hidden = 66, edge_dim = 4):
        super().__init__()
        self.n_hidden = n_hidden
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

    def forward(self, x, edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        return x, edge_attr

node_edge_embeddings = Embeddings()


edge_updates = True
batching = False
n_hidden = 66
num_gnn_layers = 2
final_dropout = 0.5
num_features = 1
edge_dim = 4


convs = nn.ModuleList()
emlps = nn.ModuleList()
batch_norms = nn.ModuleList()
for _ in range(num_gnn_layers):
    conv = GINEConv(nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden)), edge_dim=n_hidden)
    if edge_updates: emlps.append(nn.Sequential(nn.Linear(3 * n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden),))
    convs.append(conv)
    if batching:
        batch_norms.append(BatchNorm(n_hidden))
    else:
        batch_norms.append(LayerNorm(n_hidden))

mlp = nn.Sequential(Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(final_dropout), Linear(25, 2))




# party 1
src1, dst1 = party1.edge_index
x1 = embeddings_node1
edge_attr1 = embeddings_edge1

i = 0
for i in range(num_gnn_layers):
    x1 = (x1 + F.relu(batch_norms[i](convs[i](x1, party1.edge_index, edge_attr1)))) / 2
    if edge_updates:
        edge_attr1 = edge_attr1 + emlps[i](torch.cat([x1[src1], x1[dst1], edge_attr1], dim=-1)) / 2

x1 = x1[party1.edge_index.T].reshape(-1, 2 * n_hidden).relu()
x1 = torch.cat((x1, edge_attr1.view(-1, edge_attr1.shape[1])), 1)


# party 2
src2, dst2 = party2.edge_index
x2 = embeddings_node2
edge_attr2 = embeddings_edge2

i = 0
for i in range(num_gnn_layers):
    x2 = (x2 + F.relu(batch_norms[i](convs[i](x2, party2.edge_index, edge_attr2)))) / 2
    if edge_updates:
        edge_attr2 = edge_attr2 + emlps[i](torch.cat([x2[src2], x2[dst2], edge_attr2], dim=-1)) / 2

x2 = x2[party2.edge_index.T].reshape(-1, 2 * n_hidden).relu()
x2 = torch.cat((x2, edge_attr2.view(-1, edge_attr2.shape[1])), 1)





# extract edges like this
# then they need to matched with the other party, using the same indexing in a way?
# or from that parties pov, like the indices they use to send to the party that sent these
# give banks hashing functions for (others) banks or accounts?



# the values that a party would extract would then be sent to the other party, so a party
# extract values for each of the parties that it share transactions with



# code for if one wants to find the banks that another bank shares transactions with.
# might wanna implement this
"""from data.get_indices_type_data import get_indices_bdt, get_booster_data
indices_test = get_indices_bdt(df, bank = 0)
data_test = get_booster_data(parsers['data_parser'], df['regular_data'], indices_test)
test321 = data_test[0]['x']
bank_zeros_overlaps = sorted(set(test321.loc[:,['From Bank', 'To Bank']].stack()))
"""
