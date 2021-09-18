import os
import re

import pandas as pd
import matplotlib.pyplot as plt

"""
Small script that reads produced csv files, digests the available data and then plots the needed scalability graph.
Needs manual tuning for each run, can't work out-of-the-box.
"""

dir_path = '/media/CaviarBlue/Data/University/UOA_MSc/2nd_Semester/Databases/Assignements/code/obj/scala_emg_con'

csvfiles = os.listdir(dir_path)
csvfiles.sort(key=lambda f: int(re.sub('\D', '', f)))
ys = {
    'timesToGetActualData': [],
    'timesToGetUDFMapReady': [],

    'timesToCreateBST': [],
    'timesToExploreBST': [],
    'timesToCheckConstraints': [],
    'timesToPrepFails': [],

    'timesToRunRelaxation': [],
}
udf_sizes = []

for file in csvfiles:
    c_df = pd.read_csv(dir_path + '/' + file)
    for index, t in c_df.iterrows():
        try:
            x = float(t['Seconds'])
        except ValueError:
            continue

        if t['Seconds'] != 0:
            if t['Name'] == 'udf_size':
                udf_sizes.append(x)
            if t['Name'] == 'timeToGetActualData':
                ys['timesToGetActualData'].append(x)
            if t['Name'] == 'timeToGetUDFMapReady':
                ys['timesToGetUDFMapReady'].append(x)
            if t['Name'] == 'timeToCreateBST':
                ys['timesToCreateBST'].append(x)
            if t['Name'] == 'timeToExploreBST':
                ys['timesToExploreBST'].append(x)
            if t['Name'] == 'timeToCheckConstraints':
                ys['timesToCheckConstraints'].append(x)
            if t['Name'] == 'timeToPrepFails':
                ys['timesToPrepFails'].append(x)
        if t['Name'] == 'timeToRunRelaxation':
            ys['timesToRunRelaxation'].append(x)

fig, ax = plt.subplots()
ax.stackplot(udf_sizes, ys.values(), labels=ys.keys(), zorder=4,
             colors=['#FAF3DD', '#C8D5B9', '#8FC0A9', '#68B0AB', '#696D7D', '#243E36', '#086788'])
ax.legend(loc='upper left')
ax.set_xlabel('No. of Decision Variable combinations')
ax.set_ylabel("Time (sec)")
ax.set_title("Scalability of recorded durations for \nQuery Relaxation (GAS dataset)", weight='bold')
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)
plt.show()

print("adsd")
