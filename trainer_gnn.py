import copy
import numpy as np
import tqdm
from torch_geometric.nn.models.metapath2vec import sample
#from model import edge_index
from functools import partial

def train_gnn(train_data, **kwargs):

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_indices = train_data['bank_indices']
    train_data = train_data['df']

    ###################################################################
    ##### Might be come relevant to make some changes here later! #####
    ###################################################################

    # need to make dynamic here?
    num_neighbors = [100, 100]
    batch_size = 512 if train_data.num_nodes < 10000 else 128
    nn_size = len(num_neighbors)

    # loader
    transform = partial(account_for_time, main_data=train_data)
    train_loader = LinkNeighborLoader(train_data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=True, transform=None)
    #train_loader = LinkNeighborLoader(train_data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False, transform=transform)
    sample_batch = next(iter(train_loader))
    model = get_model(sample_batch, nn_size)

    # stepup
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006213266113989207)

    sample_batch.to(device)

    w_ce1 = 1.0000182882773443
    w_ce2 = 6.275014431494497
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([w_ce1, w_ce2]).to(device))

    model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, **kwargs)
    #model = train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device)

    return model


def train_homo(train_loader, train_data, train_indices, model, optimizer, loss_fn, device, index_mask = True):
    # training
    epochs = 100
    best_val_f1 = 0
    for epoch in range(epochs):

        print(f'Epoch number {epoch+1}')

        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(train_loader):
            #batch = next(iter(train_loader))
            optimizer.zero_grad()

            if index_mask:
                mask = torch.isin(batch.edge_attr[:, 0].to(torch.int), batch.input_id)
                # remove the unique edge id from the edge features, as it's no longer needed
                batch.edge_attr = batch.edge_attr[:, 1:]
                batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, index_mask = True)
                pred = out[mask]
                ground_truth = batch.y[mask]
            else:
                batch.edge_attr = batch.edge_attr[:, 1:]
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, index_mask = False)
                ground_truth = train_data.y[batch.input_id]

            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

    return model

