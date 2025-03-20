import ast
import argparse


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
