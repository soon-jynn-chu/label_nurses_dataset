import os
import pandas as pd
import numpy as np
import time
import shutil
import multiprocessing
import pickle
from datetime import timedelta, datetime
from opts import parse_opts

opt = parse_opts()

'''
Section I: Unzip Files
- unzip Stress_dataset.zip and save in Stress_dataset
- loop through each subjects in Stress_dataset and unzip the files
'''
print('Unzipping Files ...')

start_time = time.time()

new_path = opt.zip_path[:-4]

shutil.unpack_archive(opt.zip_path, new_path)

zip_list = [
    (file, sub_file)
    for file in os.listdir(new_path)
    for sub_file in os.listdir(os.path.join(new_path, file))
]


def unzip_parallel(file, sub_file):
    shutil.unpack_archive(
        os.path.join(new_path, file, sub_file),
        os.path.join(new_path, file, sub_file[:-4])
    )


pool = multiprocessing.Pool(opt.cpu_count)
results = pool.starmap(unzip_parallel, zip_list)
pool.close()

print(f"Finished in {time.time() - start_time}s")


'''
Section II: Combine files
- Store everything with the same signal in one variable
'''
print('Combining all files for each user into one ...')

start_time = time.time()

opt = parse_opts()
data_path = opt.zip_path[:-4]

signals = ['ACC', 'EDA', 'HR', 'TEMP']
acc, eda, hr, temp = None, None, None, None

names = {}
final_columns = {}

for signal in signals:
    if signal == 'ACC':
        names[f'{signal}.csv'] = ['X', 'Y', 'Z']
        final_columns[signal] = ['id', 'X', 'Y', 'Z', 'datetime']
    else:
        names[f'{signal}.csv'] = [signal]
        final_columns[signal] = ['id', f'{signal}', 'datetime']
    # Create global variable from string
    globals()[signal.lower()] = pd.DataFrame(columns=final_columns[signal])

files = [
    (file, sub_file)
    for file in os.listdir(data_path)
    for sub_file in os.listdir(os.path.join(data_path, file))
]


def process_df(df, file):
    start_timestamp = df.iloc[0, 0]
    sample_rate = df.iloc[1, 0]
    new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
    new_df['id'] = file[-2:]
    new_df['datetime'] = [(start_timestamp + i/sample_rate)
                          for i in range(len(new_df))]
    return new_df


def extract_data(file, sub_file, signal):
    if sub_file.endswith(".zip") or (signal not in names.keys()):
        return
    df = pd.read_csv(os.path.join(data_path, file, sub_file,
                     signal), names=names[signal], header=None)
    if df.empty:
        return
    return process_df(df, file)


pool = multiprocessing.Pool(opt.cpu_count)
acc = pd.concat(pool.starmap(extract_data, [i + ('ACC.csv',) for i in files]))
eda = pd.concat(pool.starmap(extract_data, [i + ('EDA.csv',) for i in files]))
hr = pd.concat(pool.starmap(extract_data, [i + ('HR.csv',) for i in files]))
temp = pd.concat(pool.starmap(
    extract_data, [i + ('TEMP.csv',) for i in files]))
pool.close()

print(f"Finished in {time.time() - start_time}s")


'''
Section III: Merge Data
- Merge all signals into one dataframe based on their timestamps
'''
print('Merging Data ...')

start_time = time.time()

ids = os.listdir(data_path)
columns = ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 'id', 'datetime']


def merge_parallel(id):
    df = pd.DataFrame(columns=columns)

    acc_id = acc[acc['id'] == id]
    eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
    hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
    temp_id = temp[temp['id'] == id].drop(['id'], axis=1)

    df = acc_id.merge(eda_id, on='datetime', how='outer')
    df = df.merge(temp_id, on='datetime', how='outer')
    df = df.merge(hr_id, on='datetime', how='outer')

    # Filling NaN values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df


pool = multiprocessing.Pool(len(ids))
df = pd.concat(pool.map(merge_parallel, ids), ignore_index=True)
pool.close()

print(f"Finished in {time.time() - start_time}s")


'''
Section IV: Label Data
- Match the timestamp from SurveyResults.xlsx with the signals
'''
print("Labelling data")

start_time = time.time()

df['datetime'] = pd.to_datetime(df['datetime'].apply(lambda x: x * (10 ** 9)))

# Read Labels File
survey_path = opt.survey_path
survey_df = pd.read_excel(survey_path, usecols=[
                          'ID', 'Start time', 'End time', 'date', 'Stress level'], dtype={'ID': str})
survey_df['Stress level'].replace('na', np.nan, inplace=True)
survey_df.dropna(inplace=True)

survey_df['Start datetime'] = pd.to_datetime(
    survey_df['date'].map(str) + ' ' + survey_df['Start time'].map(str))
survey_df['End datetime'] = pd.to_datetime(
    survey_df['date'].map(str) + ' ' + survey_df['End time'].map(str))
survey_df.drop(['Start time', 'End time', 'date'], axis=1, inplace=True)

# Convert timestamps in SurveyResults.xlsx to GMT-00:00
# Timezone where data was collected: Central Standard Time
daylight = pd.to_datetime(datetime(2020, 11, 1, 0, 0))

survey_df1 = survey_df[survey_df['End datetime'] <= daylight].copy()
survey_df1['Start datetime'] = survey_df1['Start datetime'].apply(
    lambda x: x + timedelta(hours=5))
survey_df1['End datetime'] = survey_df1['End datetime'].apply(
    lambda x: x + timedelta(hours=5))

survey_df2 = survey_df.loc[survey_df['End datetime'] > daylight].copy()
survey_df2['Start datetime'] = survey_df2['Start datetime'].apply(
    lambda x: x + timedelta(hours=6))
survey_df2['End datetime'] = survey_df2['End datetime'].apply(
    lambda x: x + timedelta(hours=6))

survey_df = pd.concat([survey_df1, survey_df2], ignore_index=True)

survey_df.reset_index(drop=True, inplace=True)

# Make file to track missing labels
open("missing_labels.txt", "x")


def parallel(id):
    new_df = pd.DataFrame(
        columns=['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 'id', 'datetime', 'label'])

    sdf = df[df['id'] == id].copy()
    survey_sdf = survey_df[survey_df['ID'] == id].copy()

    for _, survey_row in survey_sdf.iterrows():
        ssdf = sdf[(sdf['datetime'] >= survey_row['Start datetime']) & (
            sdf['datetime'] <= survey_row['End datetime'])].copy()

        if not ssdf.empty:
            ssdf['label'] = np.repeat(
                survey_row['Stress level'], len(ssdf.index))
            new_df = pd.concat([new_df, ssdf], ignore_index=True)
        else:
            with open("missing_labels.txt", "a") as myfile:
                myfile.write(
                    f"{survey_row['ID']} is missing label {survey_row['Stress level']} at {survey_row['Start datetime']} to {survey_row['End datetime']}\n")

    return new_df


pool = multiprocessing.Pool(len(ids))
results = pool.map(parallel, ids)
pool.close()
pool.join()

new_df = pd.concat(results, ignore_index=True)

print(f"Finished in {time.time() - start_time}s")


'''
Section V: Save file
'''
print('Saving ...')

with open(opt.save_file, 'wb') as file:
    pickle.dump(new_df, file, protocol=pickle.HIGHEST_PROTOCOL)

print(f'Finished in {time.time() - start_time}s')
