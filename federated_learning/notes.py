
# get_gnn --------------------------------------

# two approaches, have it inside GNN and then set model there. Or have it in manager, and then
# manager 'assigns' it to the GNN class

# to save for get_gnn

    #n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    #e_dim = (sample_batch.edge_attr.shape[1] - e_dim_adjust) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - e_dim_adjust)
    #e_dim = (sample_batch.edge_attr.shape[1]) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    #self.m_param = self.model_configs.get('params')
    #self.m_settings = utils.get_tuning_configs(self.args).get('model_settings')
    #self.batch_size = self.m_param.get('batch_size')




# GNN ----------------------------------------

# check the GINe model, just should process whole one, and then filter relevant links

# -------------------------------------------- 
# for batching for gnn
# - need to check it again, adjust to avoid copying it.
# - # need to go back and check feature engineeringen too.
# - 

