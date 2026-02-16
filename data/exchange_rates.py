"""Utility to extract exchange rates from training data and save/load to JSON."""
import os
import json
import pandas as pd
from pathlib import Path
from configs.configs import split_perc
import utils
import configs.configs as config
from data.feature_engineering import get_exchange_rates
from data.raw_data_processing import get_data


def extract_and_save(data_parser):
    """Extract exchange rates from the training split and save to JSON.

    Reads the raw CSV, computes the temporal train/vali/test split,
    extracts exchange rates from only the training rows, and saves to JSON.
    """
    df = pd.read_csv(
        f"{utils.get_data_path()}/AML_work_study/formatted_transactions_{data_parser.size}_{data_parser.ir}.csv"
    )

    # Temporarily disable normalize_currency to avoid circular recursion
    # (get_data -> load_or_extract -> extract_and_save -> get_data -> ...)
    original_flag = data_parser.normalize_currency
    data_parser.normalize_currency = False
    df, _ = get_data(df, data_parser, split_perc = split_perc)
    data_parser.normalize_currency = original_flag
    exchange_rates = get_exchange_rates(df['regular_data']['train_data']['x'])

    # Save
    save_path = _get_save_path(data_parser)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert keys to strings for JSON serialization
    rates_str = {str(k): v for k, v in exchange_rates.items()}
    save_path.write_text(json.dumps(rates_str, indent=2))

    return exchange_rates


def load(data_parser):
    """Load exchange rates from a previously saved JSON file."""
    save_path = _get_save_path(data_parser)
    with open(save_path, 'r') as f:
        rates = json.load(f)
    return {int(k): v for k, v in rates.items()}


def load_or_extract(data_parser):
    """Load exchange rates if available, otherwise extract and save."""
    save_path = _get_save_path(data_parser)
    if save_path.exists():
        return load(data_parser)
    return extract_and_save(data_parser)


def _get_save_path(data_parser):
    save_direc = os.path.join(config.save_direc_training, 'exchange_rates')
    return Path(save_direc) / f"{data_parser.size}_{data_parser.ir}.json"


def main():
    parsers = utils.parser_all()
    rates = extract_and_save(parsers['data_parser'])
    print(f"Exchange rates extracted and saved: {rates}")


if __name__ == "__main__":
    main()
