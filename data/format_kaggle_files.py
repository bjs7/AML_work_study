import numpy as np
import pandas as pd
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
import sys
import os


n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]

size = 'Small'
lr = 'HI'

inPath = '/home/nam_07/projects/AML_work_study/' + lr + '-' + size + '_Trans.csv'
outPath = os.path.dirname(inPath) + "/formatted_transactions_" + size.lower() + '_' + lr + '.csv'
#outPath = os.path.dirname(inPath) + "/formatted_transactions_test123.csv"

raw = dt.fread(inPath, columns = dt.str32)

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()
banks = dict()
tobank = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val


# ---------------------------------------------------------------------------------------
# Parse patterns file -------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

inPathPatterns = '/home/nam_07/projects/AML_work_study/' + lr + '-' + size + '_Patterns.txt'

# Pattern type mapping:
# 0: NONE (no laundering pattern, Is Laundering=0)
# 1: FAN-OUT
# 2: FAN-IN
# 3: CYCLE
# 4: GATHER-SCATTER
# 5: SCATTER-GATHER
# 6: STACK
# 7: RANDOM
# 8: BIPARTITE
# 9: UNKNOWN (Is Laundering=1 but no matching pattern in the patterns file)
PATTERN_MAP = {
    'FAN-OUT': 1,
    'FAN-IN': 2,
    'CYCLE': 3,
    'GATHER-SCATTER': 4,
    'SCATTER-GATHER': 5,
    'STACK': 6,
    'RANDOM': 7,
    'BIPARTITE': 8,
}

# Parse the patterns txt file to build a lookup dict
# Key: (Timestamp, From Bank, Account_from, To Bank, Account_to,
#        Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format) -> pattern_id
pattern_lookup = {}

current_pattern = None
with open(inPathPatterns, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('BEGIN LAUNDERING ATTEMPT - '):
            pattern_name = line.replace('BEGIN LAUNDERING ATTEMPT - ', '').split(':')[0].strip()
            current_pattern = PATTERN_MAP.get(pattern_name, 0)
        elif line.startswith('END LAUNDERING ATTEMPT'):
            current_pattern = None
        elif current_pattern is not None and line:
            parts = line.split(',')
            if len(parts) == 11:
                key = (parts[0], parts[1], parts[2], parts[3], parts[4],
                       parts[5], parts[6], parts[7], parts[8], parts[9])
                pattern_lookup[key] = current_pattern

print(f"Parsed {len(pattern_lookup)} pattern transactions from {inPathPatterns}")


# ---------------------------------------------------------------------------------------
# Format transactions -------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,From Bank,To Bank,Is Laundering,Pattern\n"

firstTs = -1
i = 0


with open(outPath, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):

        datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year
        hour = datetime_object.hour
        minute = datetime_object.minute

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

        fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

        fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"To Bank"] + raw[i,4]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(raw[i,"Amount Received"])
        amountPaidOrig = float(raw[i,"Amount Paid"])

        bank_from = get_dict_val(raw[i,"From Bank"], banks)
        bank_to = get_dict_val(raw[i,"To Bank"], banks)

        isl = int(raw[i,"Is Laundering"])

        # Look up pattern
        key = (raw[i,"Timestamp"], raw[i,"From Bank"], raw[i,2], raw[i,"To Bank"], raw[i,4],
               raw[i,"Amount Received"], raw[i,"Receiving Currency"],
               raw[i,"Amount Paid"], raw[i,"Payment Currency"], raw[i,"Payment Format"])
        pattern = pattern_lookup.get(key, 0)
        if isl == 1 and pattern == 0:
            pattern = 9

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2,amountReceivedOrig,cur1,fmt,bank_from,bank_to,isl,pattern)

        writer.write(line)


formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)





# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# add patterns here -------------------------------------------------------------------- 
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------------
# Extract the patterns first ------------------------------------------------------------
# ---------------------------------------------------------------------------------------


inPathPatterns = '/home/nam_07/projects/AML_work_study/' + lr + '-' + size + '_Patterns.txt'
outPathPatterns = os.path.dirname(inPath) + "/formatted_transactions_withpatterns_" + size.lower() + '_' + lr + '.csv'

# Pattern type mapping:
# 0: NONE (no laundering pattern, Is Laundering=0)
# 1: FAN-OUT
# 2: FAN-IN
# 3: CYCLE
# 4: GATHER-SCATTER
# 5: SCATTER-GATHER
# 6: STACK
# 7: RANDOM
# 8: BIPARTITE
# 9: UNKNOWN (Is Laundering=1 but no matching pattern in the patterns file)
PATTERN_MAP = {
    'FAN-OUT': 1,
    'FAN-IN': 2,
    'CYCLE': 3,
    'GATHER-SCATTER': 4,
    'SCATTER-GATHER': 5,
    'STACK': 6,
    'RANDOM': 7,
    'BIPARTITE': 8,
}

# Parse the patterns txt file to build a lookup dict
# Key includes amounts, currencies, and payment format for full disambiguation
# Key: (Timestamp, From Bank, Account_from, To Bank, Account_to,
#        Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format) -> pattern_id
pattern_lookup = {}

current_pattern = None
with open(inPathPatterns, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('BEGIN LAUNDERING ATTEMPT - '):
            pattern_name = line.replace('BEGIN LAUNDERING ATTEMPT - ', '').split(':')[0].strip()
            current_pattern = PATTERN_MAP.get(pattern_name, 0)
        elif line.startswith('END LAUNDERING ATTEMPT'):
            current_pattern = None
        elif current_pattern is not None and line:
            parts = line.split(',')
            if len(parts) == 11:
                key = (parts[0], parts[1], parts[2], parts[3], parts[4],
                       parts[5], parts[6], parts[7], parts[8], parts[9])
                pattern_lookup[key] = current_pattern

print(f"Parsed {len(pattern_lookup)} pattern transactions from {inPathPatterns}")




# ---------------------------------------------------------------------------------------
# Check for duplicates ------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# Re-read raw CSV and format with pattern column
raw = dt.fread(inPath, columns=dt.str32)

# Duplicate check: verify no duplicate keys in the raw CSV
print("\n--- Duplicate Check ---")
seen_keys = {}
duplicate_count = 0
for i in range(raw.nrows):
    key = (raw[i,"Timestamp"], raw[i,"From Bank"], raw[i,2], raw[i,"To Bank"], raw[i,4],
           raw[i,"Amount Received"], raw[i,"Receiving Currency"],
           raw[i,"Amount Paid"], raw[i,"Payment Currency"], raw[i,"Payment Format"])
    if key in seen_keys:
        duplicate_count += 1
        #if duplicate_count <= 5:
        print(f"  Duplicate at row {i} (first seen at row {seen_keys[key]}): {key}")
    else:
        seen_keys[key] = i

if duplicate_count > 0:
    print(f"Total duplicate keys in CSV: {duplicate_count}")
    if duplicate_count > 5:
        print(f"  (showing first 5 only)")
else:
    print("No duplicate keys found in CSV.")


# ---------------------------------------------------------------------------------------
# Format the transactions ---------------------------------------------------------------
# ---------------------------------------------------------------------------------------


currency = dict()
paymentFormat = dict()
account = dict()
banks = dict()

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,From Bank,To Bank,Is Laundering,Pattern\n"

firstTs = -1

with open(outPathPatterns, 'w') as writer:
    writer.write(header)
    for i in range(raw.nrows):

        datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
        ts = datetime_object.timestamp()
        day = datetime_object.day
        month = datetime_object.month
        year = datetime_object.year

        if firstTs == -1:
            startTime = datetime(year, month, day)
            firstTs = startTime.timestamp() - 10

        ts = ts - firstTs

        cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
        cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

        fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

        fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
        fromId = get_dict_val(fromAccIdStr, account)

        toAccIdStr = raw[i,"To Bank"] + raw[i,4]
        toId = get_dict_val(toAccIdStr, account)

        amountReceivedOrig = float(raw[i,"Amount Received"])
        amountPaidOrig = float(raw[i,"Amount Paid"])

        bank_from = get_dict_val(raw[i,"From Bank"], banks)
        bank_to = get_dict_val(raw[i,"To Bank"], banks)

        isl = int(raw[i,"Is Laundering"])

        # Look up pattern type using original string fields (including amounts for disambiguation)
        key = (raw[i,"Timestamp"], raw[i,"From Bank"], raw[i,2], raw[i,"To Bank"], raw[i,4],
               raw[i,"Amount Received"], raw[i,"Receiving Currency"],
               raw[i,"Amount Paid"], raw[i,"Payment Currency"], raw[i,"Payment Format"])
        pattern = pattern_lookup.get(key, 0)

        # If laundering but no pattern matched, assign 9 (UNKNOWN)
        if isl == 1 and pattern == 0:
            pattern = 9

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2,amountReceivedOrig,cur1,fmt,bank_from,bank_to,isl,pattern)

        writer.write(line)

formatted = dt.fread(outPathPatterns)
formatted = formatted[:,:,sort(3)]
formatted.to_csv(outPathPatterns)

print(f"Written formatted transactions with patterns to {outPathPatterns}")


# ---------------------------------------------------------------------------------------
# Sanity check --------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# Sanity check: Is Laundering vs Pattern consistency
clean_no_pattern = 0     # Is Laundering=0 and Pattern=0 (expected)
clean_with_pattern = 0   # Is Laundering=0 and Pattern!=0 (should not happen)
launder_matched = 0      # Is Laundering=1 and Pattern in 1-8 (matched a known pattern)
launder_unknown = 0      # Is Laundering=1 and Pattern=9 (no pattern in file, expected per docs)

for i in range(formatted.nrows):
    isl = formatted[i, "Is Laundering"]
    pat = formatted[i, "Pattern"]
    if isl == 0 and pat == 0:
        clean_no_pattern += 1
    elif isl == 0 and pat != 0:
        clean_with_pattern += 1
    elif isl == 1 and 1 <= pat <= 8:
        launder_matched += 1
    elif isl == 1 and pat == 9:
        launder_unknown += 1

print(f"\n--- Pattern Consistency Check ---")
print(f"Is Laundering=0, Pattern=0 (clean, no pattern):       {clean_no_pattern}")
print(f"Is Laundering=0, Pattern!=0 (clean but got pattern):  {clean_with_pattern}")
print(f"Is Laundering=1, Pattern 1-8 (matched known pattern): {launder_matched}")
print(f"Is Laundering=1, Pattern=9 (unknown/unmatched):       {launder_unknown}")

if clean_with_pattern > 0:
    print("WARNING: Some non-laundering transactions were assigned a pattern!")

# Pattern count verification: compare parsed pattern file counts vs formatted output counts
PATTERN_NAMES = {v: k for k, v in PATTERN_MAP.items()}
PATTERN_NAMES[0] = 'NONE'
PATTERN_NAMES[9] = 'UNKNOWN'

# Counts from the patterns file
parsed_counts = {}
for pat_id in pattern_lookup.values():
    parsed_counts[pat_id] = parsed_counts.get(pat_id, 0) + 1

# Counts from the formatted output
output_counts = {}
for i in range(formatted.nrows):
    pat = formatted[i, "Pattern"]
    if pat > 0:
        output_counts[pat] = output_counts.get(pat, 0) + 1

print(f"\n--- Pattern Count Verification ---")
print(f"{'Pattern':<20} {'Parsed (txt)':<15} {'Output (csv)':<15} {'Match'}")
all_match = True
for pat_id in sorted(set(list(parsed_counts.keys()) + [k for k in output_counts.keys() if k != 9])):
    p_count = parsed_counts.get(pat_id, 0)
    o_count = output_counts.get(pat_id, 0)
    match = p_count == o_count
    if not match:
        all_match = False
    print(f"{PATTERN_NAMES.get(pat_id, pat_id):<20} {p_count:<15} {o_count:<15} {'OK' if match else 'MISMATCH'}")

if 9 in output_counts:
    print(f"{'UNKNOWN':<20} {'N/A':<15} {output_counts[9]:<15} (laundering w/o pattern)")

if all_match:
    print("All pattern counts match between patterns file and formatted output.")
else:
    print("WARNING: Some pattern counts do not match!")


# ---------------------------------------------------------------------------------------
# Check for duplicates in saved csv -----------------------------------------------------
# ---------------------------------------------------------------------------------------

import utils

utils.logger_setup()
parsers = utils.parser_all()
parsers['data_parser'].testing = True
utils.set_seed(parsers['data_parser'].seed, True)
# -------------

parsers['data_parser'].ibm_fe = True
parsers['data_parser'].ibm_hp = True
parsers['data_parser'].train_for_final = True
parsers['data_parser'].add_ids = False

#parsers['fl_parser'].fl_algo = 'FedVert'
parsers['fl_parser'].fl_algo = 'full_info'
parsers['data_parser'].scenario = 'full_info'

df = pd.read_csv(f"{utils.get_data_path()}/AML_work_study/formatted_transactions_withpatterns_{parsers['data_parser'].size}_{parsers['data_parser'].ir}.csv")


df.duplicated().any()

df[df['Is Laundering'] == 1]









"""

#bank_from = get_dict_val(int(raw[i, "From Bank"]), frombank)
#bank_to = get_dict_val(int(raw[i, "To Bank"]), frombank)



df1 = pd.read_csv('/home/nam_07/projects/AML_work_study/formatted_transactions_small_HI.csv')
df2 = pd.read_csv('/home/nam_07/projects/AML_work_study/formatted_transactions_test123.csv')


l1 = list(df1['From Bank']) + list(df1['To Bank'])
len(set(l1))

l2 = list(df2['From Bank']) + list(df2['To Bank'])
len(set(l2))


df1.loc[:,'From Bank']
df1.loc[:,'To Bank']

test123 = list(df1.loc[:,'From Bank']) + list(df1.loc[:,'To Bank'])
len(set(test123))

len(set(list(set(df1.loc[:,'From Bank'])) + list(set(df1.loc[:,'To Bank']))))

len(list(set(test123)))

len(list(set(df1.loc[:,'From Bank'])))
len(list(set(df1.loc[:,'To Bank'])))

len(list(set(list(set(df1.loc[:,'From Bank'])) + list(set(df1.loc[:,'To Bank'])))))


for i in range(raw[:,1].shape[0]):

    if raw[i,1] == '0014':
        break
        #test123.append(raw[i,1])

for j in range(raw[:,1].shape[0]):

    if raw[j,1] == '014':
        break
        #test123.append(raw[i,1])



test123 = []
test1234 = []

for i in range(raw[:,1].shape[0]):

    if raw[i,1][0] == '0':
        test123.append(raw[i,1])

    if raw[i,1][0:2] == '00':
        test1234.append(raw[i,1])




t1, t2 = [], []
for i in test123:
    if i == '014':
        t1.append(i)

for i in test1234:
    if i == '0014':
        t2.append(i)



'014' in test123
'0014' in test123

int('014')
int('0014')

'014' in test1234
'0014' in test1234

len(test123)
len(test1234)

t1, t2 = [], []
for i in test123:
    t1.append(int(i))

for i in test1234:
    t2.append(int(i))

t1 = set(t1)
t2 = set(t2)


#bank_from = int(raw[i,"From Bank"])
#bank_to = int(raw[i,"To Bank"])


"""

