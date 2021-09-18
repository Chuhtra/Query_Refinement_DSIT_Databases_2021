from time import time
from zipfile import ZipFile

import pandas as pd
import re

import psycopg2.extras
import argparse
import os
import csv

# dir_with_csv_files = '/media/CaviarBlue/Data/Downloads/db_datasets'
db_name1 = 'emg'
db_name2 = 'gas'

"""
Script that handles downloaded datasets and prepares needed Databases in PostgreSQL.
"""


def main():
    t1 = time()
    dir_path = parseArguments()

    try:
        pass
        data_dirs = unzipData(dir_path)  # ['temp_emg', 'temp_gas']#
    except IsADirectoryError:
        data_dirs = []
        print("\nUnzipping failed: Please place the zip files in an empty directory.")
        quit(1)

    conn, cur = postgresConnect('postgres')
    cur.execute(f"""CREATE DATABASE {db_name1};""")
    cur.execute(f"""CREATE DATABASE {db_name2};""")
    disconnectFromDatabase(conn, cur)

    for i, d in enumerate(data_dirs):
        print(f"Preparing dataset {i + 1}...")
        if d.__contains__(db_name1):
            fixEMGcsv(f'{dir_path}/{d}/', 'myo_data.csv')
            print("Adding dataset 1 to database...")
            pass
            conn, cur = postgresConnect(db_name1)
            cur.execute(f"""
                       CREATE TABLE emg_data(
                                       timestamp bigint NOT NULL,
                                       emg1 integer NOT NULL,
                                       emg2 integer NOT NULL,
                                       emg3 integer NOT NULL,
                                       emg4 integer NOT NULL,
                                       emg5 integer NOT NULL,
                                       emg6 integer NOT NULL,
                                       emg7 integer NOT NULL,
                                       emg8 integer NOT NULL
                       );
                       """)
            cur.execute(f"""COPY emg_data FROM '{dir_path}/{d}/new_myo_data.csv' csv HEADER;""")
            cur.execute(f"""ALTER TABLE emg_data ADD COLUMN time_id BIGSERIAL PRIMARY KEY;""")
            disconnectFromDatabase(conn, cur)
        elif d.__contains__(db_name2):
            concatenateCSVfiles(dir_path + '/' + d, "gas_data.csv")
            print("Adding dataset 2 to database...")
            pass
            conn, cur = postgresConnect(db_name2)
            cur.execute(f"""
                       CREATE TABLE gas_data(
                                       Time real NOT NULL,
                                       CO real NOT NULL,
                                       Humidity real NOT NULL,
                                       Temperature real NOT NULL,
                                       Flow_rate real NOT NULL,
                                       Heater_voltage real NOT NULL,
                                       R1 real NOT NULL,
                                       R2 real NOT NULL,
                                       R3 real NOT NULL,
                                       R4 real NOT NULL,
                                       R5 real NOT NULL,
                                       R6 real NOT NULL,
                                       R7 real NOT NULL,
                                       R8 real NOT NULL,
                                       R9 real NOT NULL,
                                       R10 real NOT NULL,
                                       R11 real NOT NULL,
                                       R12 real NOT NULL,
                                       R13 real NOT NULL,
                                       R14 real NOT NULL,
                                       Date bigint NOT NULL
                        );
                        """)

            cur.execute(f"""COPY gas_data FROM '{dir_path}/{d}/gas_data.csv' csv HEADER;""")
            cur.execute(f"""ALTER TABLE gas_data ADD COLUMN time_id BIGSERIAL PRIMARY KEY;""")

            disconnectFromDatabase(conn, cur)

    print(f"Databases are created. Elapsed time: {time() - t1} sec.")


def parseArguments():
    parser = argparse.ArgumentParser(description="Prepare Postgres for Query Refinement ")

    parser.add_argument("--filepath",
                        nargs="?",
                        required=True,
                        help="Absolute path where the directory the downloaded zip files are located.")

    args = parser.parse_args()

    return args.filepath


def unzipData(dir_path):
    db_names = []
    print("Extracting zip files...")
    zipfiles = os.listdir(dir_path)

    for i, zipfile in enumerate(zipfiles):
        # Create a ZipFile Object and load sample.zip in it
        with ZipFile(dir_path + '/' + zipfile, 'r') as zipObj:
            # Get a list of all archived file names from the zip
            listOfFileNames = [x for x in zipObj.namelist() if x.endswith('.csv') and not x.__contains__('/')]

            n = f'temp_{db_name1 if len(listOfFileNames) == 1 else db_name2}'
            # Only the emg dataset contains a single .csv file
            db_names.append(n)
            zipObj.extractall(members=listOfFileNames, path=dir_path + f'/{n}')

    print("Zip files extracted.")
    return db_names


def concatenateCSVfiles(dir_path, final_file):
    print("Please wait...")
    csvfiles = os.listdir(dir_path)

    # First we get the column names and add one more
    with open(dir_path + '/' + csvfiles[0], 'r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        fieldnames.append("Date")

    # We initialize the unified table with pandas
    my_df = pd.DataFrame(columns=fieldnames)

    # We append every csv to the initialized table. The added column takes the values from the file name.
    for c in csvfiles:
        c_df = pd.read_csv(dir_path + '/' + c)
        name = c.replace('.csv', '')
        name = name.replace('_', '')
        new_column = [name for _ in range(0, c_df.shape[0])]
        c_df['Date'] = new_column
        my_df = pd.concat([my_df, c_df])

    # We sort the completed unified table based on datetime.
    my_df.sort_values(['Date', 'Time (s)'], ascending=[True, True])

    # Separate csv files are deleted.
    for c in csvfiles:
        os.remove(dir_path + '/' + c)

    # We fix the column names of the unified table.
    for col in fieldnames:
        my_df = my_df.rename(columns={col: re.sub(r' \(.+\)', '', col)})

    # We export it to disk.
    my_df.to_csv(dir_path + '/' + final_file, index=False)


def fixEMGcsv(path, name):
    """
    The EMG csv file contains 3 lines that shouldn't exist, so we remove them.
    """
    with open(path + name, 'r') as inp, open(path + 'new_' + name, 'w') as out:
        writer = csv.writer(out)
        for i, row in enumerate(csv.reader(inp)):
            if i > 2:
                writer.writerow(row)
    os.remove(path + name)
    pass


def postgresConnect(db_name):
    connection = psycopg2.connect(f"dbname='{db_name}' user='postgres' host='localhost'")
    # Open a cursor to perform database operations
    db_handle = connection.cursor()
    connection.set_session(autocommit=True)  # To avoid transactions idling open.

    return connection, db_handle


def disconnectFromDatabase(conn, cur):
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
