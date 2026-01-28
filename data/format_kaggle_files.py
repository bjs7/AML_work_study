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

header = "EdgeID,from_id,to_id,Timestamp,\
Amount Sent,Sent Currency,Amount Received,Received Currency,\
Payment Format,From Bank,To Bank,Is Laundering\n"

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

        line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d,%d,%d\n' % \
                    (i,fromId,toId,ts,amountPaidOrig,cur2,amountReceivedOrig,cur1,fmt,bank_from,bank_to,isl)

        writer.write(line)


formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)











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

