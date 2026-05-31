"""Global training constants shared across all FL algorithms."""

from configs.paths import get_data_path

split_perc = [0.60, 0.20]
epochs = 100

save_direc_training = get_data_path() + '/AML_work_study/experiments/'
