from db_preparation import postgresConnect, disconnectFromDatabase, db_name1, db_name2

"""
Simple script that cleans the created databases.
"""

conn, cur = postgresConnect('postgres')

cur.execute(f"DROP DATABASE IF EXISTS {db_name1};")
cur.execute(f"DROP DATABASE IF EXISTS {db_name2};")

disconnectFromDatabase(conn, cur)

print("Databases are removed.")
