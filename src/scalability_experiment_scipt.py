import copy

from client import syntacticQueryParsing, connectToDatabase, disconnectFromDatabase
from server import execute_query
from LibraryOfTime import timekeeper, startTime, saveTimers, LibraryOfTime

"""
Script to create data for the scalability experiments. Needs fine-tuning with each run, don't use out-of-the-box.
"""

startTime()
q = '''
    SELECT time_id, offset IN_DOMAIN [1, 10], [5, 10]
    FROM gas_data.flow_rate
    WHERE avg_amp() in [50, 200] MAX
        and        max_amp_excess_left(4) in [-2, 0]  MAX
        and        max_amp_excess_right(4) in [-2, 0]  MAX
    LIMIT REFINED 50
    '''
query_dict = syntacticQueryParsing(q)

if query_dict['table'].__contains__('emg'):
    db_name = 'emg'
else:
    db_name = 'gas'

conn, cur = connectToDatabase(db_name)

query = f"SELECT MIN(time_id), MAX(time_id) FROM {(query_dict['table'])};"
cur.execute(query)
bounds = list(cur.fetchone())

vars = query_dict['decision_variables']
i = 0
v = True
while query_dict['domains'][0][1] <= bounds[1]:

    print(query_dict)
    _ = execute_query(cur, copy.deepcopy(query_dict), plot=False)
    saveTimers(timekeeper, f"../obj/scala/{str(i)}.csv")
    timekeeper.purge()
    i += 1

    if query_dict['domains'][1][1] >= bounds[1]:
        v = 0
    elif query_dict['domains'][0][1] >= bounds[1]:
        v = 1
    else:
        v = not v

    query_dict['domains'][v][1] += 20

disconnectFromDatabase(conn, cur)
