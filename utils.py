import ast
import argparse
import logging
import os
import sys
import numpy as np
import torch
import random

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger("MyApp")

def logger_setup():
    # Setup logging
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )

def parse_banks(value):

    value = value.strip()

    if value.startswith('[') and value.endswith(']') and ':' in value:
        start, end = value[1:-1].split(':')
        return list(range(int(start), int(end)))

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError('Invalid format')


def parse_data_split(value):
    
    value = value.strip()

    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, list) and all(isinstance(i, (int, float)) for i in parsed_value):
            return parsed_value
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid format for --data_split. Use [0.6, 0.2]")


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")

