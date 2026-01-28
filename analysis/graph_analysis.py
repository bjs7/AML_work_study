# %%

import pandas as pd
import utils
import igraph as ig
import numpy as np

from data.raw_data_processing import get_data
from configs.configs import split_perc
from collections import defaultdict


# make a "test" settings function or something like that 


# %%

utils.logger_setup()
parsers = utils.parser_all()

df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")



# %%

num_nodes = max(df[['from_id', 'to_id']].stack())
edges_to_add = [[df.loc[i,'from_id'], df.loc[i,'to_id']] for i in range(df.shape[0])]
#clusters_to_add = [[df.loc[i,'From Bank'], df.loc[i,'To Bank']] for i in range(df.shape[0])]
g = ig.Graph(n=num_nodes, edges = edges_to_add, directed=True)



# %%

#isolated_nodes = [v.index for v in g.vs if v.degree() == 0]
#g.delete_vertices([v for v in g.vs if v.degree() == 0])

weak_components = g.connected_components(mode='weak')
len(weak_components)
weak_components.sizes()

component_nodes = defaultdict(list)
for node_idx, component_id in enumerate(weak_components.membership):
    component_nodes[component_id].append(node_idx)



# %%

df_from = df[['from_id', 'From Bank']].rename(columns={'from_id': 'account', 'From Bank': 'bank'})
df_to = df[['to_id', 'To Bank']].rename(columns={'to_id': 'account', 'To Bank': 'bank'})
stacked_df = pd.concat([df_from, df_to], ignore_index=True)
stacked_df.drop_duplicates(subset='account', keep='first')


for row in stacked_df.iterrows():
    g.vs[row[1]['account']]['cluster'] = row[1]['bank']



# %%

membership = g.vs['cluster']
contracted_g = g.copy()  # Make a copy first
contracted_g.contract_vertices(membership, combine_attrs=None)
contracted_g.simplify(combine_edges='sum')



# %%

weak_components = contracted_g.connected_components(mode='weak')
len(weak_components)
weak_components.sizes()
