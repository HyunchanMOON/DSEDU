import json, os, glob
import pandas as pd
import matplotlib.pyplot as plt

def read_json_file(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        json_object = json.load(f)

    return json_object

files = glob.glob(os.path.join('./JSON/', '*.json'))
files.sort()

df = pd.DataFrame()
for i, f in enumerate(files):
    j_to_dic = read_json_file(f)
    tmp =  pd.DataFrame.from_dict([j_to_dic], orient='columns')
    df = tmp if i ==0 else pd.concat([df, tmp], axis=0)

print(df.head())
df.plot(x='datetime', y=['avgSpeed', 'maxSpeed','avgRPM','maxRPM','avgCoolantTemp','avgIntakeTemp', 'avgDriveTime'], subplots=True, figsize=(15,8))
# df.plot(x='datetime', y='avgRPM')

plt.savefig('all_graph.jpg')
plt.show()
