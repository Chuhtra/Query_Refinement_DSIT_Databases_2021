import copy
from time import time
from texttable import Texttable
from LibraryOfTime import timekeeper, startTime, saveTimers

import random as r

import ast
import psycopg2.extras
import server


def main():
    # First we get user input regarding the dataset she wants to try.
    startTime()
    db_name = getUserDBPref()
    t1 = time()

    # We then connect to the db.
    conn, cur = connectToDatabase(db_name)
    timekeeper.timeToConnectDB = time() - t1

    # We print a readout of the dataset's columns
    getDBStats(cur)

    # We then read the query from file.
    t1 = time()
    try:
        with open("query.txt", "r+") as file:
            query = ''.join(file.readlines())

    except FileNotFoundError:
        earlyDeath("\n Please provide a query.txt file in src directory.")
    timekeeper.timeToReadQuery = time() - t1

    print("\n Query was read from file successfully...")
    print(" Requested query is: \n")
    print(query)

    # Next, we validate the syntax of the query to avoid crashes (some cases may have slipped away :') )
    t1 = time()
    query_dict = syntacticQueryParsing(query)
    timekeeper.timeToValidateSyntax = time() - t1

    if query_dict["table"] != db_name + "_data":
        earlyDeath("\n The chosen dataset and the query do not match. Try again.")

    print(" \n Query is being processed... \n")

    # If all is good, we execute the query to get the results
    t1 = time()
    results = server.execute_query(cur, copy.deepcopy(query_dict), plot=True)
    timekeeper.timeToExecuteQuery = time() - t1

    # We store recorded timings to be able to later use for experimental results.
    query_dict.pop('decision_variables')  # I do this to avoid File name too long errors.
    saveTimers(timekeeper, f"../obj/{str(list(query_dict.values()))}_TIMES_{str(r.randint(1, 1000))}.csv")

    # print(f"Initial time: {timekeeper.getInitialTime()} sec.")
    # print(f"Elapsed time: {timekeeper.getExecutionTime()} sec.")
    # print(f"Total Execution time (after query proc.): {timekeeper.timeToExecuteQuery} sec.")
    # print(f"Total time (Recorded parts): {timekeeper.getInitialTime() + timekeeper.getExecutionTime()} sec.")

    # We close the db connection.
    disconnectFromDatabase(conn, cur)


def connectToDatabase(db_name):
    """
    Function that encapsulates the database connection returning the connection itself and a cursor for querying.

    :param db_name: The database name, selected by user.
    :return: The connection and the database cursor objects.
    """
    try:
        # Connect to an existing database
        connection = psycopg2.connect(f"dbname={db_name} user='postgres' host='localhost'")
        print(f"\n Connected to {db_name} Database.")
    except psycopg2.OperationalError:
        connection = None
        earlyDeath("Unable to connect to the database.")

    # Open a cursor to perform database operations
    db_handle = connection.cursor()
    # psycopg2.extras.register_hstore(db_handle)  # To be able to pass python dictionaries to PL/Python functions.
    connection.set_session(autocommit=True)  # To avoid transactions idling open.

    return connection, db_handle


def disconnectFromDatabase(conn, cur):
    """
    Close communication with the database.

    :param conn: The database connection object.
    :param cur: The database cursor object.
    """
    cur.close()
    conn.close()
    print(f"\n Disconnected from database.")


def getUserDBPref():
    """
    Simple function to get user preference on dataset selection.
    Instead of input() the wanted char can be hardcoded to have an immediate run.

    :return: Three-letter word that is the name of the selected database.
    """
    print("\nPlease enter a number to select a database: \n"
          "1: emg - Electromyography Signal Data \n"
          "2: gas - Voltage Signal ")
    while True:
        x = input()
        if x == '1':
            return 'emg'
        elif x == '2':
            return 'gas'
        else:
            print("Please choose a correct number")
    pass


def getDBStats(cur):
    """
    Function that prints the selected datasets table column info in a nice tabular format.

    :param cur: The database cursor used for querying.
    """

    cur.execute("""SELECT table_name 
                    FROM information_schema.tables
                    WHERE table_schema = 'public';""")
    table = cur.fetchone()[0]

    cur.execute(f"""SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table}';""")
    column_info = list(map(list, cur.fetchall()))  # We fetch the rows and transform them to a list of lists.

    tex = Texttable()
    tex.add_rows([[f"Column Name", "Datatype"]] + [[k[0], k[1]] for k in column_info])

    print("\n Database Info")
    print(tex.draw())
    print("\n")


def syntacticQueryParsing(query):
    """
    This function processes the string query and returns it in an organised dictionary format. The syntax analysis is
    based on an automatic tokenization of the string using the sqlparse library, and a subsequent token digestion based
    on key words of the query. This custom digestion was needed as the input is not a normal SQL query but a CP one.

    :param query: String value containing the user query.
    :return: Dictionary object containing the query in an organized manner.
    """
    import re
    domain_template = "^\\[(-?\\d+|None), *(-?\\d+|None)]$"
    available_functions = ['avg_amp', 'max_amp_excess_right', 'max_amp_excess_left']
    con_template = f"^({'|'.join(available_functions)})\\(-?\\d*\\)$"  # For more numeric args use \((-?\d+, *) *\)$

    dic = {'decision_variables': [],
           'domains': [],
           'table': None,
           'table_column': None,

           'cp_constraints': [],  # max_amp_excess_right
           'constraint_args': [],
           'constraint_limits': [],
           'constraint_targets': [],  # minimization

           'cardinality': None,
           'refined': None
           }

    def getTokens(x):
        """
        This function tokenizes the string, and returns the useful ones in a list.
        :param x: The string query.
        :return: A list of string tokens.
        """
        from sqlparse import parse

        parsed = parse(x)[0].tokens
        tokens_list = [str(tok) for tok in parsed if (str(tok) != ' ' and str(tok) != '\n')]

        return tokens_list

    def syntaxCheck(template, values):
        """
        This function compares the syntax of the strings items in a list, with a given regex template. Before making the
        comparison, it makes sure that the items are indeed strings.

        :param template: A regular expression with which the items should match.
        :param values: A list of items that should be matched with a regex template.
        :return: True or false, depending on matching of all items.
        """
        return all([re.search(template, x if isinstance(x, str) else str(x)) for x in values])

    tokens = getTokens(query)
    try:
        for i, token in enumerate(tokens):  # We iterate over every token, and digest information when keywords are met.
            if token == 'SELECT':
                if i != 0:
                    earlyDeath("Error at SELECT keyword")
                else:
                    if tokens[i + 1] != 'time_id, offset':  # Checking var names is more than syntactic check but anyway
                        earlyDeath("ERROR at SELECT variables")
                    else:
                        dic['decision_variables'] = [x for x in tokens[i + 1].split(', ')]

            elif token.startswith('IN_DOMAIN'):
                if i != 2:
                    earlyDeath("Error at IN_DOMAIN keyword")
                else:
                    dec_domains = ast.literal_eval('[' + token.replace('IN_DOMAIN ', '') + ']')
                    if len(dec_domains) != 2:
                        earlyDeath("ERROR at decision variable domains")
                    elif syntaxCheck(domain_template, dec_domains):
                        dic['domains'] = [x for x in dec_domains]
                    else:
                        earlyDeath("ERROR at decision variable domains")

            elif token == 'FROM':
                if i != 3:
                    earlyDeath("Error at FROM keyword")
                else:
                    tab = tokens[i + 1].split('.')
                    if len(tab) != 2:
                        earlyDeath("ERROR at table and column, only one of each acceptable.")
                    else:
                        dic['table'] = tab[0]
                        dic['table_column'] = tab[1]

            elif token.startswith('WHERE'):
                if i != 5:
                    earlyDeath("Error at WHERE keyword")
                else:
                    w_tokens = token.split()
                    for j, w_token in enumerate(w_tokens):
                        if w_token == 'WHERE' or w_token == 'and' and w_tokens[j + 2] == 'in':
                            con_name = w_tokens[j + 1]
                            con_dom = w_tokens[j + 3] + w_tokens[j + 4]
                            con_goal = w_tokens[j + 5]

                            if syntaxCheck(con_template, [con_name]):
                                dic['cp_constraints'].append(re.sub(r'(\d+|\(|\)|,)', '', con_name))  # only letters
                                x = re.sub(r'([A-Za-z]+|\s+|\(|\)|_)', '', con_name).split(',')  # only numbers
                                if x != ['']:
                                    x = [ast.literal_eval(x) for x in x]
                                else:
                                    x = [None]
                                dic['constraint_args'].append(x)
                            else:
                                earlyDeath("ERROR at constraint variable name and arguments")

                            if syntaxCheck(domain_template, [con_dom]):
                                dic['constraint_limits'].append(ast.literal_eval(con_dom))
                            else:
                                earlyDeath("ERROR at constraint variable domains")

                            if con_goal == 'MAX':
                                dic['constraint_targets'].append('maximization')
                            elif con_goal == 'MIN':
                                dic['constraint_targets'].append('minimization')
                            else:
                                earlyDeath("Error at constraint target keyword")

            elif token == 'LIMIT':
                if i != 6:
                    earlyDeath("Error at LIMIT keyword")
                else:
                    if tokens[i + 1] == 'REFINED':
                        dic['refined'] = True
                        dic['cardinality'] = ast.literal_eval(tokens[i + 2])
                    else:
                        dic['refined'] = False
                        dic['cardinality'] = ast.literal_eval(tokens[i + 1])

    except SyntaxError:
        earlyDeath("Program crashed due to SyntaxError in parsing query file.")
    except IndexError:
        earlyDeath("Program crashed due to IndexError in parsing query file.")

    if len(dic['cp_constraints']) <= 0 or len(dic['constraint_limits']) <= 0:
        # Messing with line structure is caught here.
        earlyDeath("ERROR please fix query structure.")

    if dic['refined'] is None:
        dic['refined'] = False

    return dic


def earlyDeath(s):
    """
    Simple function to print error message and then quit with error code 1.

    :param s: Message to print.
    """
    print('\n ERROR: ' + s)
    quit(1)


if __name__ == "__main__":
    main()
