from superseded import gnn
import superseded.utils as utils
import logging
import argparse
import pandas as pd
from models.base import Model
from models_ import booster
from data.raw_data_processing import get_data
from configs.configs import split_perc
import superseded.save_load_models as slm
from mix.relevant_banks import get_relevant_banks


# need to pass the "relevant" to the bank or not. Every bank should be considered in fedvertical case?
# need to pass the "relevant" to the bank or not. Every bank should be considered in fedvertical case?
# need to pass the "relevant" to the bank or not. Every bank should be considered in fedvertical case?



#parsers['fl_parser'].fl_algo = 'full_info'
#parsers['data_parser'].scenario = 'individual_banks' if parsers['fl_parser'].fl_algo != 'full_info' else 'full_info'

#parsers['fl_parser'].fl_algo = 'individual'






# remeber to adjust the amount of seeds

# need more loggin on federated avg

# probably also want more data on hwo each bank perferms individually
# like their f1 and then be able to get average f1 over them all etc.

# also for the vertical case. What to do about something like the mean/std for the standardization of data?
# somehow obtain a global one?
# also why exchanging everything to dollars is important potentially

# fed average I should modify it such that the weight to each depends on the amount of data they have

# for FL there are two approach? Batch individually, calculate one layer for each batch,
# send info to other banks, get info back, update batches etc.
# Second, which is probably much better, no batching, just calculate info, send out,
# get info back update etc.

# IF NON-BATCHING IS USED, CHANGE THE BATCHNORM TO LAYERNORM IN GNN MODEL!

# have logging for most, but still need for federated manager

# caching in managers?

# ALSO NEED TO ADD MORE LOGGING AND SAVE THE LOGGING IN THE EXPERIEMENTS FOLDER!

# smart way for how to handle / sort folders on hpc and how to set "algos" etc. in the .sh file

# be sure that gpu is used on hpc

# in tuning, most of the metrics calcuations can be skipped
# conditino that only adds max_prob, avg_prob etc. if not full info?

# NEED TO DOUBLE CHECK feature_engi_graph_data, not sure it copies, applies, selects the right dataframe

# probably need to update the inference up_laundering_values function to match batching

# currently just have one seed in full_info, just as right now stuff is just being tested if it runs corrctly

# double check the update nodes etc. like the parts when full graph data set is sliced into bank subsets

# log or save data on how many banks has a f1 of 0, or close to 0 and the amount of data they have access to
# and f1 score of all the banks

# batch vs no batching? Make analysis of banks amount of data, and their amount of 1 observations they see



# residuals i gnn model?

# IMPLEMENT FINETUNING? LIKE FOR THE DATASETS MEDIUM OR HIGHER, THERE I SHOULD USE
# THE PARAMETERS FOUND FOR THE SMALL SIZED DATASET? LIKE TRAIN ON SMALL DATASET, AND USE THIS TO TUNE
# MEDIUM/LARGE DATA SET MODELS? THEY DO THAT IN THE PAPER!

# Ask claude about this, like how to implement the no tuning? Like what is the best approach
# Use the split indices or where should one set those settings and conditions?


# Check GFP parameters igen? Fra "vertex_stats_cols": [3,4], til "vertex_stats_cols": [3],

# https://claude.ai/chat/5aae28e8-c824-483f-8049-dde365855f0b
# https://claude.ai/chat/07d74b0d-bca1-47d1-b8b3-870d26950a8f - clustering / HPC stuff
# https://claude.ai/chat/0d3f2e34-b685-48b2-a137-eb3fb61bf50a - Tuning approaches
# https://claude.ai/chat/5aae28e8-c824-483f-8049-dde365855f0b fl framwork


# tuning? Use the one from the testing on full data? All run tunning on the FL?
# so far the code is written such that tuning is done between the banks, so once that is done, 
# they need to make feature engineering after again


# one can argue that data is somewhat similar, or like from similar distribution, and there
# no or limited heterogenity in data is present and therefore one can use same hyperparamters?

# tænker også over, lad os lige tuning localt, hvordan ville det virke i praksis? Ville man
# teste optimere mange gange og så lave en gradient? Ville man test mange parameters
# og så se hvilken en man kommer tættest med? Sender man alle forskellige værdier af parameters
#  til manager? 
#  How? Both for GNN og booster? Hvordan ville det virke for dem?
# Gøre det løbende med FL? Eller køre FL færdigt og så tune på model selv efter?
# Men vise de rtunes på modellen lokalt efter, hvad så når man træner med train og vali data
# global? Så har man ændret parameters lokalt? Og de er forskellige fra de globale, som
# måske eller højst sandsynlige er blevet tuned?

# hierarchical of global/local parameters? Some global, some local?

# papers with methods for implementation
# https://ieeexplore.ieee.org/document/9440789
# https://arxiv.org/pdf/1712.01887
# https://arxiv.org/pdf/1602.05629

# python inherit stuff
# https://claude.ai/chat/f3f2a491-47f7-46cd-8203-45bee93d102b


# onehot encoding
# https://chatgpt.com/c/68cc667e-053c-8330-af41-b707bf27d7a7


# https://chatgpt.com/c/68434731-9634-8012-904c-c7e7856e4f04
# https://chatgpt.com/c/68442d4a-4398-8012-9670-b91e7034c591



# for now only learnable parameters are being shared, using non-learnable
# can potentially be done once encryption is started to being applied


# if reg or graph epochs is used. Or also is for decision trees, yes?
# just update in one, and then another for sending to manager?




#





def main():


    utils.logger_setup()
    parser = utils.get_parser()
    args = parser.parse_args()


    # initiate





    # training of model -------------------------------------------------------------------------------------------------------------
        # number of rounds







    # Model inference -------------------------------------------------------------------------------------------------------------

    






    return 0






if __name__ == '__main__':
    main()






