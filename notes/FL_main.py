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






