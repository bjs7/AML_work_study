# %%

# packages
import sys
sys.path.append('/home/nam_07/projects/AML_work_study/AML_work_study')
import pandas as pd
import utils
import copy
import numpy as np
import matplotlib.pyplot as plt
import data.feature_engi as fe


# %%
parsers = utils.parser_all()
fl_parser, data_parser, gnn_parser = parsers['fl_parser'], parsers['data_parser'], parsers['gnn_parser']
parsers['data_parser'].testing = True
df = pd.read_csv(utils.get_data_path() + '/AML_work_study/formatted_transactions' + f'_{data_parser.size}' + f'_{data_parser.ir}' + '.csv')


# %%
df

# %%
# check for duplicates
df.duplicated().any()

# check for nulls
df.isnull().any().any()

# Nothing found. Though this is expected as the paper also states this.


# %%

# First create some inspections plots

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df['Amount Sent'], bins=100, edgecolor='black')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Amount Sent')

axes[0, 1].hist(df['Amount Received'], bins=100, edgecolor='black')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Amount Received')

axes[1, 0].hist(df['Sent Currency'], bins=100, edgecolor='black')
axes[1, 0].set_xlabel('Sent Currency')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Histogram')

axes[1, 1].hist(df['Received Currency'], bins=100, edgecolor='black')
axes[1, 1].set_xlabel('Received Currency')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()

# The plot shows that the columns 'Amount Sent' and 'Amount Received' are both clearly skewed
# This can also be confirmed by the summary statistics of the two

summary = df[['Amount Sent', "Amount Received"]].describe().T
summary = summary.round(3)  # Round to 3 decimals
print(summary.to_string(float_format='%.3f'))


# %%

#Check with log frequency
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].hist(df['Amount Sent'], bins=100, edgecolor='black', log = True)
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Amount Sent')

axes[1].hist(df['Amount Received'], bins=100, edgecolor='black', log = True)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Amount Received')

plt.tight_layout()
plt.show()

# %%

# Plot raw amounts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(['Amount Sent', 'Amount Received']):
    axes[i].hist(df[col], bins=100, edgecolor='black')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{col} (Raw)')
plt.tight_layout()
plt.show()

# %%

# Log transform and plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(['Amount Sent', 'Amount Received']):
    log_vals = np.log(df[col])
    axes[i].hist(log_vals, bins=100, edgecolor='black')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{col} (log)')
plt.tight_layout()
plt.show()

# %%

# Log1p transform and plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(['Amount Sent', 'Amount Received']):
    log1p_vals = np.log1p(df[col])
    axes[i].hist(log1p_vals, bins=100, edgecolor='black')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{col} (log1p)')
plt.tight_layout()
plt.show()

# %%

# Standardize the log-transformed values and plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(['Amount Sent', 'Amount Received']):
    log_vals = np.log(df[col])
    log_std = (log_vals - log_vals.mean()) / log_vals.std()
    axes[0, i].hist(log_std, bins=100, edgecolor='black')
    axes[0, i].set_xlabel('Value')
    axes[0, i].set_ylabel('Frequency')
    axes[0, i].set_title(f'{col} (log + standardized)')

    log1p_vals = np.log1p(df[col])
    log1p_std = (log1p_vals - log1p_vals.mean()) / log1p_vals.std()
    axes[1, i].hist(log1p_std, bins=100, edgecolor='black')
    axes[1, i].set_xlabel('Value')
    axes[1, i].set_ylabel('Frequency')
    axes[1, i].set_title(f'{col} (log1p + standardized)')
plt.tight_layout()
plt.show()


# %%

exchange_rates = fe.get_exchange_rates(df)
df = fe.normalize_amounts(df, exchange_rates)


# %%

#Check amounts again, after they haven been normalized
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].hist(df['Amount_Sent_Normalized'], bins=100, edgecolor='black')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Amount_Sent_Normalized')

axes[1].hist(df['Amount_Received_Normalized'], bins=100, edgecolor='black')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Amount_Received_Normalized')

plt.tight_layout()
plt.show()

# Still highly skewed. Similar can be seen in the summary statistics

summary = df[['Amount_Sent_Normalized', "Amount_Received_Normalized"]].describe().T
summary = summary.round(3)  # Round to 3 decimals
print(summary.to_string(float_format='%.3f'))



# %%

#Check with log frequency
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].hist(df['Amount_Sent_Normalized'], bins=100, edgecolor='black', log = True)
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Amount_Sent_Normalized')

axes[1].hist(df['Amount_Received_Normalized'], bins=100, edgecolor='black', log = True)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Amount_Received_Normalized')

plt.tight_layout()
plt.show()


# %%

# Apply log transformations to the amount values
df = fe.log_transformer(df)

# %% 

#Check with log frequency
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].hist(df['Amount_Sent_Normalized_Log'], bins=100, edgecolor='black')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Amount_Sent_Normalized_Log')

axes[1].hist(df['Amount_Received_Normalized_Log'], bins=100, edgecolor='black')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Amount_Received_Normalized_Log')

plt.tight_layout()
plt.show()

# %%

df = fe.universal_features_restructure(df)

# %%

df



# %%
