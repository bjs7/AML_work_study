import utils
import logging
import argparse
import pandas as pd
from models.base import Model
from models import booster, gnn
from data.raw_data_processing import get_data
from configs.configs import split_perc
import inference_saving.save_load_models as slm
from relevant_banks import get_relevant_banks


import FL_message as FLm


# Check GFP parameters igen? Fra "vertex_stats_cols": [3,4], til "vertex_stats_cols": [3],

# https://claude.ai/chat/5aae28e8-c824-483f-8049-dde365855f0b
# https://docs.python.org/3/library/typing.html

# https://claude.ai/chat/d0c63d5e-ee22-4323-bc37-e60479b85147 - parallel calculations
# https://claude.ai/chat/b6a87c36-1365-4a4d-85fa-3adca1cd5fd0 - python inherit
# https://claude.ai/chat/07d74b0d-bca1-47d1-b8b3-870d26950a8f - clustering / HPC stuff


# https://claude.ai/chat/0d3f2e34-b685-48b2-a137-eb3fb61bf50a - Tuning approaches
# https://claude.ai/chat/f53275d6-6647-4eb5-9eb0-2410f3840fb4 - tuning approaches

# https://claude.ai/chat/5aae28e8-c824-483f-8049-dde365855f0b fl framwork


# Things that needs to be done:
# - Wanna do it such that calculations can be done in parallel
# - All feature engineering happens individually
# - Create classes etc. for message receiving/passing?


# Banks make calculations, updates model or sends message to eachother
# The receive a message which is the new model or messages used to update their model

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

# needs to prep data


# https://chatgpt.com/c/68434731-9634-8012-904c-c7e7856e4f04
# https://chatgpt.com/c/68442d4a-4398-8012-9670-b91e7034c591



# Need to check nodes/edge sampling again. Need to be more sure on this


# Though one can probably run full sample for all the banks, it might be worth
# to consider to apply sampling to the larger once, just to potentially replicate more realistic
# settings, and see the impact.


# ----------------------------------------------------------------------
# CHECK FOR DUBLICATES? 
# NEED TO MAKE MORE FEATURE ENGINEERING? LIKE CHECK TIMESTAMP PATTERNS?
# ----------------------------------------------------------------------

# CRITICAL ERROR!!!
# NEED TO CHECK DOUBLE CHECK THE VALEUS OF RECEIVING AMOUNT/CURRENCY!
# THERE ARE VARIATIONS IN THESE ON SOME TRANSACTIONS! BOTH AMOUNT AND CURRENCY!


# Maybe also keep timestamp? It is at least relevant for hetereo graphs. So maybe just keep for all?
# Though questionable what the point is for homo-grahps, because then it just becomes a feature that
# increases in value over time
# Like timestamp could potentially allow the GNN to see time between transactions
# that might be suspicious, and see if that is relevant. Then also use it in decision tree and other models?


# Also still find some of the stuff in the AML paper, or at least the code in the github
# a bit strange. For example how they standardize the edge features.


# Might wanna go back to "simple" GNN's with just 3 inputs, and then filter in them after?
# If making changes here, make sure that it is calculates correctly



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






