from time import time
from LibraryOfTime import timekeeper
from bst_class import Tree, TreeNode, Failpoint, NodeVariable

import copy
import pickle
import random as r
import numpy as np


# TODO : There are parameters that could be defined by user instead of defaults (search for 'User defined' comments)
# TODO : I couldn't run a query that provides fails during 1st relaxation run (search for 'UNTESTED CODE' comments)

def execute_query(db_cursor, query_dict, plot=False):
    """
    This method is the highest abstraction point in the query processing. It prepares query dictionary, retrieves data
    in the UDF map, explores the Binary Search Tree and initiates query relaxing if needed. Alternatively, the acquired
    results are enough.

    :param db_cursor: The database connection cursor to query data.
    :param query_dict: The user query organized in dictionary format.
    :param plot: Boolean flag for plotting the query results or not.
    :return: The acquired results, with or without refinement.
    """

    t1 = time()
    # We remove None decision variable domains
    # if all(isinstance(value, str) for value in query_dict.values()):
    #     query_dict = fixDictionaryValues(query_dict)
    query_dict_f = fixDecisionDomains(db_cursor, query_dict)
    timekeeper.timeToFixDictionary = time() - t1

    # We retrieve UDF map (and data for plotting)
    udf_map, data = getUDFmap(db_cursor, query_dict_f)

    # ONLY USEFUL FOR SCALABILITY EXPERIMENT
    timekeeper.udf_size = len(udf_map) - 2

    t1 = time()
    # We initialize a BST based on user query
    tree = Tree(
        TreeNode(
            query_dict['decision_variables'],
            query_dict['domains']
        ),
        query_dict['cp_constraints'],
        query_dict['constraint_limits'],
        query_dict['constraint_targets']
    )
    timekeeper.timeToCreateBST = time() - t1

    print(f"\n {'PRELIMINARY' if query_dict_f['refined'] else ''} QUERY RESULTS: \n")
    k = query_dict_f['cardinality']

    t1 = time()
    bst, fail_points, unrefined_results = exploreBSTree(tree, udf_map, k, initial_run=True,
                                                        refinement_on=query_dict_f['refined'])
    timekeeper.timeToExploreBST = time() - t1

    found_results_num = len(unrefined_results)

    if query_dict_f['refined']:
        print(f"\n REQUESTED CARDINALITY: {k}. FOUND: {found_results_num} results.")
    else:
        if k is None:
            k = +np.inf
        print(f"\n FOUND: {found_results_num} results. "
              f"{'Try enabling query refinement for better results.' if found_results_num < k else ''}")

    if query_dict_f['refined'] is False or found_results_num == k:  # If query is satisfied
        final_results = unrefined_results
    elif found_results_num < k and query_dict_f['refined']:  # If we need query relaxing
        print('\n Not enough results found. Query relaxation is initiating for extended search...')
        t1 = time()
        relaxed_results = query_relaxing(bst, fail_points, udf_map, k - found_results_num)
        timekeeper.timeToRunRelaxation = time() - t1

        final_results = unrefined_results + relaxed_results
    else:  # Unreachable section
        final_results = []

    if plot is True:  # If plotting was requested an image file is stored to disk.
        fig = plotDiagram(data, final_results, query_dict['table_column'])
        query_dict.pop('decision_variables')  # I do this to avoid File name too long errors.
        fig.savefig(f"../obj/{str(list(query_dict.values()))}.png")
        print(f"\n Check obj directory for image named {str(list(query_dict.values()))}.")

    # UNCOMMENT TO GET INFORMATION ON CONSTRAINTS WHEN LOOKING FOR QUERIES TO RUN EXPERIMENTS
    # for p in final_results:
    #     tid = getTimeIdIndex_Binary(p)
    #     print(udf_map[f"{p[tid].value}+{p[not tid].value}"])

    # UNCOMMENT TO RUN THE MANUAL QUERY TIGHTENING CASE
    # print("Simulated result ranking at client will run.")
    # k = 50 #None
    # if k is None or k==np.inf:
    #     print("Simulated result ranking at client FAILED because specific k value wasn't added in code.")
    #     quit(1)
    # t1 = time()
    # _, final_results = getResultRanking(final_results, bst.constraints, udf_map, k)
    # timekeeper.timeToRankAtClient = time() - t1

    return final_results


####################################################################################################################
#                                       Query Dictionary Preparation Functions
####################################################################################################################

# def fixDictionaryValues(query_dict):
#     """
#     This method returns the dictionary values to their normal data types. They get passed as strings.
#      Maybe not useful after move to PL/Python.
#     """
#     for key in query_dict:
#         t = query_dict[key]
#         try:
#             query_dict[key] = ast.literal_eval(t)
#         except ValueError:
#             continue
#     return query_dict


def fixDecisionDomains(db_cursor, query_dict):
    """
    We check the domains of the given decision variables. If they contain None values we try to fix them based on min
    and max values of the respective table column, supposing that they exist.

    :param db_cursor: The database connection cursor to query data.
    :param query_dict: The user query organized in dictionary format.
    :return: The user query organized in dictionary format, with possible None domain values fixed.
    """
    # Get column names to match user's SELECT variables with
    # Maybe unnecessary given that queries got restrained to two mandatory SELECT variables.
    db_cursor.execute(f"""SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{query_dict['table']}';""")
    column_info = list(map(''.join, db_cursor.fetchall()))

    for i, var in enumerate((query_dict['decision_variables'])):
        try:  # Search requested variable in table
            _ = column_info.index(var)
        except ValueError:  # variable doesn't belong to the table, thus can't get fixed with some column.
            continue

        # If found, get the user defined domain.
        varmin = (query_dict['domains'])[i][0]
        varmax = (query_dict['domains'])[i][1]

        # If domain contains Nones then the respective column's edge value must be retrieved from the table
        variables = []
        if varmin is None: variables.append(f"MIN({var})")
        if varmax is None: variables.append(f"MAX({var})")

        if len(variables) != 0:
            query = f"SELECT {', '.join(variables) + ' '} FROM {(query_dict['table'])};"
            db_cursor.execute(query)
            bounds = list(db_cursor.fetchone())  # We fetch only one cause we know a single row is returned anyway.

            # Iterate over retrieved values to change Nones. Min always comes first.
            while len(bounds) != 0:
                if varmin is None:
                    varmin = bounds[0]
                    query_dict['domains'][i][0] = bounds[0]
                    del bounds[0]
                if varmax is None:
                    varmax = bounds[0]
                    query_dict['domains'][i][1] = bounds[0]
                    del bounds[0]

        # Offset is a non-table variable that is bound to time_id, so we check for its domains here with time_id.
        if var == 'time_id' and 'offset' in query_dict['decision_variables']:
            offset_i = query_dict['decision_variables'].index('offset')

            if query_dict['domains'][offset_i][0] is None: query_dict['domains'][offset_i][0] = 1
            if query_dict['domains'][offset_i][1] is None:
                if len(bounds) != 2:
                    query = f"SELECT MIN(time_id), MAX(time_id) FROM {(query_dict['table'])};"
                    db_cursor.execute(query)
                    bounds = list(db_cursor.fetchone())
                query_dict['domains'][offset_i][1] = bounds[1] - bounds[0] + 1

    return query_dict


####################################################################################################################
#                                                UDF values' map Functions
####################################################################################################################

def getData(db_cursor, query_dict):
    """
    Function that encloses the only querying of actual data to the database.

    :param db_cursor: The database connection cursor to query data.
    :param query_dict: The user query organized in dictionary format.
    :return: Retrieved data as a numpy array.
    """
    time_id_index = getTimeIdIndex_Binary(query_dict['decision_variables'])
    initial_query = f"""SELECT time_id AS x, {query_dict['table_column']} AS y
                        FROM {query_dict['table']} AS t
                        WHERE {query_dict['domains'][time_id_index][0]} <= time_id AND
                            {query_dict['domains'][time_id_index][1] + query_dict['domains'][not time_id_index][1]} 
                                >= time_id;
                    """  # Second constraint is x_max+lx_max
    db_cursor.execute(initial_query)
    return np.array(db_cursor.fetchall())


def checkDataVariability(data_rows):
    v = np.var(data_rows[:,1])
    if v <= 5e-28:
        print("\n Chosen segment presents no variability. Please select a different segment.")
        quit(1)
    pass


def getUDFmap(db_cursor, query_dict):
    """
    Function that prepares the UDF_values map which is utilized instead of a synopsis-based pipeline like Searchlight.
    It starts with the querying of the raw data of a table column given time domain limits.

    :param db_cursor: The database connection cursor to query data.
    :param query_dict: The user query organized in dictionary format.
    :return: The UDF map and retrieved data.
    """
    '''
    First we load the data for the requested decision variables' domain(s).
    '''
    t1 = time()
    data_rows = getData(db_cursor, query_dict)
    timekeeper.timeToGetActualData = time() - t1

    checkDataVariability(data_rows)

    '''
    For the given row set, we calculate the UDF values as the paper considers the constraints "readily available".
    build_map flag is used to skip lengthy calculations, if we already have done them in the past.
    '''
    build_map = True

    t1 = time()
    if build_map:
        udf_map = build_UDF_map(data_rows,
                                query_dict['cp_constraints'],
                                query_dict['constraint_args'],
                                query_dict['domains'][0],
                                query_dict['domains'][1])
    else:
        def readUdfMap(name):
            """
            Small function that reads a UDF map from file.

            :param name: The filename. CAUTION: THE FILENAME MUST BE MANUALLY CHANGED WITH EACH RUN.
            :return: The UDF map.
            """
            with open(f"../obj/{name}.pkl", 'rb') as handle:
                data = handle.read()
            return pickle.loads(data)

        udf_map = readUdfMap("['time_id', 'offset']_time_id_in_[1, None]_offset_in_[5, 8]_359")

    timekeeper.timeToGetUDFMapReady = time() - t1

    if timekeeper.timeToGetUDFMapReady > 300:  # If creating this map takes more than 5 minutes, store it as well.
        def saveUDF2File(udf, dic):
            """
            This helper method is used to save the UDF results as they may take hours to be computed while not being
            different with each run. The query dictionary is also stored in a text file with corresponding name,
            to keep things sane. A custom name is used with a random element, to help with similar query runs. The
            corresponding query dictionary is also saved.

            :param udf: The data structure that holds the values that correspond to each result pair's constraints.
            :param dic: The user query organized in dictionary format.
            """
            # Save the UDF results
            f = open(f"../obj/{str(dic.values())}_UDFMAP_{str(r.randint(1, 1000))}.pkl", "wb")
            pickle.dump(udf, f)
            f.close()

            # Save the corresponding query
            import json
            exDict = {'exDict': dic}
            with open(f'../obj/{str(dic.values())}_DICT.txt', 'w') as file:
                file.write(json.dumps(exDict))  # use `json.loads` to do the reverse

        saveUDF2File(udf_map, query_dict)

    return udf_map, data_rows


def build_UDF_map(data, constraints, constraint_args, x_domain, lx_domain):
    """
    The function that iterates over data pair values and retrieves the constraint values of the map.

    :param data: The data queried from the database.
    :param constraints: The list of constraints that need to be recorded.
    :param constraint_args: The arguments for each constraint.
    :param x_domain: The domain of the first decision variable (time_id)
    :param lx_domain: The domain of the second decision variable (offset)
    :return: The UDF map.
    """
    # For domains only [] case is supported. Paper also uses [] i think.
    lx_range = range(lx_domain[0], lx_domain[1] + 1)
    x_range = range(x_domain[0], x_domain[1] + 1)

    def initializeUDFMap(constraints_list):
        """
        Initialization function that sets min and max values that will be fixed later in the loop, for each constraint.

        :param constraints_list: The list of requested constraints
        :return: A dictionary initialized with inf min and max values for each constraint.
        """
        dic = {"mins": {}, "maxs": {}}  # In mins/maxs we will hold the edge values of each constraint in the region.
        for con in constraints_list:
            dic['mins'][con] = +np.inf
            dic['maxs'][con] = -np.inf
        return dic

    udf_results = initializeUDFMap(constraints)
    for i, x in enumerate(x_range):
        for j, lx in enumerate(lx_range):

            if x + lx > x_domain[1] + lx_domain[1]:  # Check if we are out of bounds in the time-series
                continue
            else:
                udf_results[f'{x}+{lx}'] = {}

            for c, constraint in enumerate(constraints):
                result = None
                if constraint == 'avg_amp':
                    # I get the needed indexes of the data table, for the pair of decision values, so I can slice it.
                    start = int(np.where(data[:, 0] == x)[0])
                    end = int(np.where(data[:, 0] == x + lx)[0]) + 1
                    result = avg_amp(data[start:end, 1])
                    udf_results[f'{x}+{lx}'][constraint] = result

                elif constraint == 'max_amp_excess_right':
                    result = max_amp_excess_right(data, x, lx, constraint_args[c])
                    udf_results[f'{x}+{lx}'][constraint] = result

                elif constraint == 'max_amp_excess_left':
                    result = max_amp_excess_left(data, x, lx, constraint_args[c])
                    udf_results[f'{x}+{lx}'][constraint] = result

                # I also need to store the min-max values of the constraints.
                if result < udf_results['mins'][constraint]:
                    udf_results['mins'][constraint] = result
                if result > udf_results['maxs'][constraint]:
                    udf_results['maxs'][constraint] = result

    return udf_results


# def getUDFvalue(key, constraint):
#     try:
#         return udf_map[key][constraint.name]
#     except KeyError:  # Fast fix, used for (essentially) out of bounds errors.
#         continue

####################################################################################################################
#                                                Binary Search Functions
####################################################################################################################

def exploreBSTree(tree, udf_map, k, initial_run=None, refinement_on=False, prevRPvars=None, results_unreached=True):
    """
    Binary Search Tree (BST) is being built and explored in DFS fashion. The branching is happening by randomly
    selecting an unbound variable and dissecting its domain. When a variable is bounded it's no-longer checked in
    children nodes. Constraint failures or complete Variable bounding lead to backtracking. Failure points and leaf
    results are saved, respectively, to be returned in the end of the search.

    :param results_unreached: Boolean flag showing if the requested search cardinality has been met, in the current
    search or a previous one. Useful when a recorded failure contains only one node and relaxation hasn't reached at
    least k results.
    :param prevRPvars: Useful for relaxation. Dictionary containing constraints and VC values of the previous tree, for
    the current root node.
    :param k: The cardinality requested for the Binary Tree search.
    :param refinement_on: Boolean flag showing if refinement has been requested.
    :param initial_run: Boolean flag showing if the current search is the initial search requested by the user.
    :param tree: The tree that has been instantiated for this search.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :return: The explored tree, the recorded fail points and the recorded results.
    """

    tightening_on = False  # This can't be something else in the beginning
    relaxing_on = not initial_run and not tightening_on

    initial_results = []
    failpoints = []

    while True:
        if not tree.current_node.visited:

            t1 = time()
            # Constraint check
            constraints_passed, VCs, failed_constraints = \
                checkConstraints(udf_map,
                                 tree.current_node.unbound_variables + tree.current_node.variables, tree.constraints,
                                 tightening_on, MRK=tree.MRK if tightening_on else None)
            timekeeper.timeToCheckConstraints += (time() - t1)

            if constraints_passed:

                if len(tree.current_node.unbound_variables) > 0:  # Not leaf: We can explore deeper in the subtree.
                    child1, child2 = tree.branchOut()

                    # New tree pointer
                    tree.current_node.visited = True
                    tree.current_node = child1

                else:  # We got a leaf, with successful constraints(aka a result)!
                    tree.current_node.visited = True

                    if refinement_on and initial_run and len(initial_results) >= k:
                        # we are on original query run, but must tighten it.

                        tightening_on = True
                        initial_run = False
                        results_unreached = False
                        print('\n More valid results than requested have been found.\n Please wait for results\'s '
                              'ranking, for query tightening.Ignore preliminary results.\n')

                        tree.MRK, initial_results = getResultRanking(initial_results, tree.constraints, udf_map, k)

                    if refinement_on and tightening_on:  # if tightening is active
                        RK = rankResult(tree.current_node.variables, tree.constraints, udf_map)
                        if RK > tree.MRK:  # Store the leaf only if it is a good enough one.
                            initial_results.append((tree.current_node.variables, RK))
                            initial_results = sorted(initial_results, key=lambda result: result[1], reverse=True)[:k]
                            tree.MRK = initial_results[-1][1]

                    elif refinement_on and relaxing_on:  # we are on relaxation run
                        RP = resultPenalty(tree.current_node.variables, prevRPvars['old_constraints'], udf_map,
                                           prevRPvars['rootVCs'])
                        append_flag = False
                        update_MRP_flag = False

                        if len(initial_results) < k:  # we have less than k results
                            append_flag = True

                            if len(initial_results) + 1 == k or \
                                    (tree.current_node == tree.root and not results_unreached):
                                # we reach k or subtree results not enough to reach k, so we need to recalculate MRP
                                update_MRP_flag = True

                        elif len(initial_results) == k:
                            if RP < tree.MRP:  # Store the leaf only if it is a good enough one.
                                append_flag, update_MRP_flag = True, True

                        # The actual storing and MRP updating
                        if append_flag:
                            initial_results.append((tree.current_node.variables, RP))
                        if update_MRP_flag:
                            initial_results = sortPenalties(initial_results, k)
                            tree.MRP = initial_results[-1][1]

                    else:  # we are just on original query run
                        if refinement_on is False and len(initial_results) == k:
                            break  # This is the simple LIMIT case

                        initial_results.append(tree.current_node.variables)

                        # For interactivity, initial results are printed.
                        #
                        # It should NOT immediately print for refinement if not initial run.
                        # E.g. on relaxing, subtree may have more than the missing results as eligible for relaxed
                        # constraints and the best ones will be checked after the meh ones have been printed.
                        printVarPairs(tree.current_node.variables)

                    tree.backtrack()

            else:  # Constraints failed.
                if refinement_on and tightening_on is False:  # If fail points should be recorded.
                    t1 = time()
                    node, maximal_bounds = prepare_failpoint(tree.current_node, udf_map, tree.constraints, VCs,
                                                             tree.MRP)
                    if node is not None: failpoints.append(Failpoint(node, maximal_bounds, failed_constraints, VCs))
                    timekeeper.timeToPrepFails += (time() - t1)

                tree.backtrack()

        if tree.current_node == tree.root:  # We have returned to the root thus the entire tree is searched.
            break

    if tightening_on:  # Tightening got activated, thus the final result set must be printed.

        print("TIGHTENED QUERY RESULTS: \n")
        for res in initial_results:
            printVarPairs(res[0])

        # Here we remove the ranking from each result, as it's no longer needed.
        initial_results = [x[0] for x in initial_results]

    return tree, failpoints, initial_results


def checkConstraints(udf_map, decision_variables, constraints, tightening_on, MRK):
    """
    We iterate over all decision variables' combinations in search for some combination that satisfies every constraint.
    This means that some later leaf in the sub-tree will be a desirable result. If the constraints are never met
    completely then we return False. Excessive search over the values takes place, in case constraints aren't monotonic.

    Important: In this function we presuppose that udf_map has keys of 'time_id+offset' form and that these 2 are our
    only decision variables.

    :param tightening_on: Boolean flag showing whether query tightening has been activated.
    :param MRK: Used when query tightening is active, it's the Minimum Rank that the subtree needs to surpass. When
    query tightening is False, MRK is set to None to be essentially ignored.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :param decision_variables: The variables used to acquire the UDF_MAP key.
    :param constraints: The constraints that need to be checked.
    :return: Either True, or False along with VC values of the nodes and a list with the failed constraints.
    """

    time_id_range, offset_index_range = getLoopRanges(decision_variables)

    VCs = {}  # Ratio of violated constraints, used later for query relaxation.
    failed_constraints = set()

    for i in time_id_range:
        for j in offset_index_range:
            key = f"{i}+{j}"
            results = []

            for constraint in constraints:
                try:
                    t_ = udf_map[key][constraint.name]
                except KeyError:  # Fast fix, used for (essentially) out of bounds errors.
                    continue

                # If constraint is somewhere unbound, we use t value to get a satisfiable outcome due to equality.
                c_min = constraint.min if constraint.min is not None else t_
                c_max = constraint.max if constraint.max is not None else t_

                if c_min <= t_ <= c_max:  # Actual constraint check
                    results.append(True)
                else:
                    failed_constraints.add(constraint.name)
                    results.append(False)
                    # break # Breaking here brings the optimization they tested for lazy computation. (Not implemented)

            if len(results) == 0:  # there are no constraints, so it's an automatic pass
                return True, None, None

            VCs[key] = (len(results) - np.sum(results)) / len(results)

            if all(results):  # If all constraints are satisfied, it means the node passes the constraints' check.
                if tightening_on:  # If tightening is on, an extra constraint is added that can disqualify a result.
                    RK = rankResult([NodeVariable('time_id', i, i), NodeVariable('offset', j, j)],
                                    constraints, udf_map)
                    if RK > MRK:  # Instead of utilizing paper's BRK, we only search for an RK that is good enough.
                        return True, None, None
                else:
                    return True, None, None

    return False, VCs, failed_constraints


def prepare_failpoint(node, udf_map, constraints, VCs, MRP):
    """
    Here we get the a' and b' noted in the paper, as well as the Relaxation Penalties. Similar structure to the
    constraints_check loops. We calculate Relaxation Distances and Penalties as described in sections 3.1 and 4.1 of
    the paper.

    Constraint relaxation based on MRP isn't implemented cause it doesn't make sense to me. In the paper example the
    constraint ends up tightened not tightly relaxed. This non-implementation is the case for custom RP function anyway.

    :param MRP: Maximum Relaxation Penalty, metric used for subtree pruning.
    :param VCs: Ratio of violated constraints, used later for query relaxation.
    :param node: The node which is the failure point.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :param constraints: The constraints, since we need to calculate a' and b'.
    :return: If passed the BRP-MRP dynamic constraint, we return the fail node and the a'-b' pair, else None
    """
    maximal_bounds = {}
    decision_variables = node.variables + node.unbound_variables

    time_id_range, offset_index_range = getLoopRanges(decision_variables)

    for constraint in constraints:
        maximal_bounds[constraint.name] = {'max': -np.inf, 'min': +np.inf}  # These are the a' and b' of the paper.

    BRP, WRP = +np.inf, -np.inf
    for i in time_id_range:
        for j in offset_index_range:
            key = f"{i}+{j}"
            for constraint in constraints:
                try:
                    score = udf_map[key][constraint.name]
                except KeyError:  # Fast fix, used for (essentially) out of bounds errors.
                    continue

                # a' and b'
                if score > maximal_bounds[constraint.name]['max']:
                    maximal_bounds[constraint.name]['max'] = score
                if score < maximal_bounds[constraint.name]['min']:
                    maximal_bounds[constraint.name]['min'] = score

            RP = resultPenalty(key, constraints, udf_map, VCs)  # Total Relaxation Penalty, calculated to find BRP/WRP.

            if RP > WRP: WRP = RP
            if RP < BRP: BRP = RP

    if BRP > MRP:
        return None
    else:
        node.BRP, node.WRP = BRP, WRP
        return node, maximal_bounds


####################################################################################################################
#                                                Relaxation Functions
####################################################################################################################

def constraintRelaxationDistance(t_, a, b, min_fc, max_fc):
    """
    Implementation of the Relaxation Distance function documented in the paper.

    :param t_: The outcome of a result for a specific constraint.
    :param a: The minimum boundary of a specific constraint.
    :param b: The maximum boundary of a specific constraint.
    :param min_fc: The assigned min function the constraint should take.
    :param max_fc: The assigned max function the constraint should take.
    :return: Relaxation Distance for a single constraint.
    """
    if a is None: a = t_
    if b is None: b = t_

    if a <= t_ <= b:
        return 0
    elif t_ > b:
        return (t_ - b) / (max_fc - b)
    elif t_ < a:
        return (a - t_) / (a - min_fc)


def resultPenalty(candidate, constraints, udf_map, VCs=None):
    """
    Calculation of total result Relaxation Penalty, function documented in paper.

    :param candidate: The candidate result. It may be a string or a list, depending on where the function is called.
    :param constraints: THe list of constraints for which Relaxation distance must be calculated.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :param VCs: The dictionary with the precalculated VC values.
    :return: The total Relaxation Penalty for a given result.
    """
    if isinstance(candidate, str):
        key = candidate
    else:
        time_id_index = False if candidate[0].name == "time_id" else True
        key = f"{candidate[time_id_index].value}+{candidate[not time_id_index].value}"

    RD = -np.inf
    for constraint in constraints:

        RD_c = constraintRelaxationDistance(t_=udf_map[key][constraint.name], a=constraint.min, b=constraint.max,
                                            min_fc=udf_map['mins'][constraint.name],
                                            max_fc=udf_map['maxs'][constraint.name])
        w_c = 1  # User defined, per constraint weight with 1 as default.
        if RD < w_c * RD_c:
            RD = w_c * RD_c

    a = 0.5  # User defined parameter to determine preference between RD and VC.
    RP = a * RD + (1 - a) * VCs[key]  # Total Relaxation Penalty
    return RP


def getRelaxationPenalties(results, k):
    """
    Helper function that sorts a list of results, finds the MRP and returns both.
    :param results: A list of candidate results.
    :param k: The requested cardinality.
    :return: Updated MRP and sorted-updated results
    """
    penalized_results = sortPenalties(results, k)
    MRP = penalized_results[-1][1]  # Maximum Penalty
    return MRP, penalized_results


def sortPenalties(penalized_results, k=None):
    """
    Helper function that sorts results based on PENALTY, and keeps only k of them.
    :param penalized_results: A list of candidate results.
    :param k: The requested cardinality.
    :return: Sorted-updated results
    """
    if k is None:
        k = len(penalized_results)
    return sorted(penalized_results, key=lambda result: result[1])[:k]


def query_relaxing(bst, fail_points, udf_map, k_new):
    """
    Function called after original query has completed with less results than requested. For each recorded fail point
    a replay is initiated. If this relaxation step produces new failpoints, these are recorded to be replayed if the
    cardinality isn't met from the first set of fail points.

    Important: The case of a failpoint search producing new fail points hasn't been tested, as no query could be found
    to simulate such behavior.

    :param bst: The Binary Search Tree that was explored in the initial query search.
    :param fail_points: The list of fail points recorded in the initial query search.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :param k_new: The cardinality that is required after the initial query search has completed.
    :return: The relaxed results that can make the query meet the requested cardinality.
    """
    results = []
    prev_fails = [(fail_points, bst)]
    while k_new > len(results) and len(prev_fails) > 0:  # Each loop is a step to a lower "search level" on the tree.
        nr, nf = [], []
        for fails, tree in prev_fails:
            new_results, new_failures = initiateReplaying(tree, fails, udf_map, k_new)
            nr += new_results
            if len(new_failures) != 0:
                nf += new_failures  # UNTESTED CODE

        prev_fails.clear()
        prev_fails = nf

        results = results + sortPenalties(nr, k_new - len(results))

    if len(results) < k_new: print("\n Requested cardinality is too big for given decision variable domains.")

    print(f"\n RELAXED QUERY RESULTS (top-{len(results)}): \n")
    for res in results:
        printVarPairs(res[0])

    # We return only the variables, discarding the RP values.
    return [x[0] for x in results]


def initiateReplaying(bst, failpoints_list, udf_map, k_new):
    """
    This method is the iteration of a set of fail points, initiating a Tree Search for each one. Each one is becoming
    the root node of a new search tree, and the results of all of those trees are competing for a place in the list.
    Before initiating the search, the failed constraints are relaxed maximally. The failpoints are sorted based on BRP
    and therefore the loop can terminate before complete iteration.

    :param bst: The Binary Search Tree that was explored in the initial query search.
    :param failpoints_list: The list of fail points recorded in the initial query search.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :param k_new: The cardinality that is required after the initial query search has completed.
    :return: The relaxed results for this level of relaxation, along with possible recorded fail points.
    """
    bst_constraints = copy.deepcopy(bst.constraints)
    prevRPvars = {'old_constraints': copy.deepcopy(bst.constraints), 'rootVCs': None}  # Variables of the previous tree.

    failpoints_list = sorted(failpoints_list, key=lambda failpoint: failpoint.node.BRP)
    new_results = []
    new_failures = []
    for fail in failpoints_list:
        if fail.node.BRP >= bst.MRP: break  # We stop the loop since the failpoints_list list is sorted on BRP (asc).
        for constraint in bst_constraints:
            if constraint.name in fail.failed_constraints:
                constraint.max = fail.maximal_bounds[constraint.name]['max']
                constraint.min = fail.maximal_bounds[constraint.name]['min']

        results_unreached = True if len(new_results) < k_new else False
        tree = Tree(fail.node, bst_constraints)
        prevRPvars['rootVCs'] = fail.VCs
        new_bst, new_fail_points, results = exploreBSTree(tree, udf_map, k_new, initial_run=False,
                                                          refinement_on=True, prevRPvars=prevRPvars,
                                                          results_unreached=results_unreached)

        new_results = sortPenalties(new_results + results, k_new)

        if len(new_fail_points) != 0:
            new_failures.append((new_fail_points, new_bst))  # UNTESTED CODE

        if bst.MRP > new_bst.MRP:
            bst.MRP = new_bst.MRP

    return new_results, new_failures


####################################################################################################################
#                                                Tightening Functions
####################################################################################################################

def constraintRankingFunction(t_, a, b, target):
    """
    Implementation of the Ranking function documented in the paper.

    :param t_: The outcome of a result for a specific constraint.
    :param a: The minimum boundary of a specific constraint.
    :param b: The maximum boundary of a specific constraint.
    :param target: The target (minimization/maximization) requested for a specific constraint.
    :return: Result Ranking for a single constraint.
    """
    if target == 'maximization':
        return (b - t_) / (b - a)
    elif target == 'minimization':
        return (a - t_) / (b - a)


def rankResult(candidate, constraints, udf_map):
    """
    Calculation of total result Ranking, function documented in paper.

    :param candidate: The candidate result
    :param constraints: The list of constraints for which RK must be calculated.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :return: The total RanK for a given result.
    """
    w_c = 1 / len(constraints)  # constraint weights [0,1] and sum to 1. Can be user defined, respecting these rules.

    time_id_index = False if candidate[0].name == "time_id" else True
    key = f"{candidate[time_id_index].value}+{candidate[not time_id_index].value}"

    rank_sum = 0
    for constraint in constraints:
        RK_c = constraintRankingFunction(udf_map[key][constraint.name], constraint.min, constraint.max,
                                         constraint.target)
        rank_sum += w_c * RK_c

    RK = 1 - rank_sum  # Full RanK of result.
    return RK


def getResultRanking(results, constraints, udf_map, k):
    """
    Function that retrieves the ranking for a set of results, sorts them, keeps the k best and gets the Minimum Rank.

    :param results: A list of candidates.
    :param constraints: The list of NodeVariable constraints for which RK must be calculated.
    :param udf_map: The data structure that holds the values that correspond to each result pair's constraints.
    :param k: The requested cardinality.
    :return: Minimum RanK and sorted-updated list of candidates.
    """
    ranked_results = []
    for candidate in results:
        RK = rankResult(candidate, constraints, udf_map)
        ranked_results.append((candidate, RK))

    ranked_results = sorted(ranked_results, key=lambda result: result[1], reverse=True)[:k]
    MRK = ranked_results[-1][1]  # Minimum RanK
    return MRK, ranked_results


####################################################################################################################
#                                                Helper Functions
####################################################################################################################


def getTimeIdIndex_Binary(decision_variables):
    """
    Helper function that gets the list of decision variables (in our case always 2 variables) and returns the index of
    time_id variable.

    :param decision_variables: A list of 2 variables, one of which must be time_id
    :return: A boolean value representing indexes 0 or 1.
    """
    if isinstance(decision_variables[0], str):
        name = decision_variables[0]
    else:
        name = decision_variables[0].name
    return False if name == "time_id" else True


def getIterationSet(variable):
    """
    Function that receives a NodeVariable and returns a range object for iteration. If variable is bounded then a list
    with a single value is returned.

    :param variable: The variable we need to get the range from.
    :return: An iterable range or single-item list.
    """
    if variable.value is not None:
        return [variable.value]
    else:
        return range(variable.min, variable.max + 1)


def getLoopRanges(decision_variables):
    """
    Function that returns two iterable objects, one for each decision variable.

    :param decision_variables: The variables we need to get the range from.
    :return: 2 objects.Can be iterable ranges, single-item lists or one of each.
    """
    # Boolean values as index to flip 1 and 0.
    time_id_index = getTimeIdIndex_Binary(decision_variables)
    time_id_iteration = getIterationSet(decision_variables[time_id_index])
    offset_index_iteration = getIterationSet(decision_variables[not time_id_index])

    return time_id_iteration, offset_index_iteration


def printVarPairs(variable_list):
    """
    Simple function that prints query results. Variable content is first sorted to avoid chaotic printing.
    :param variable_list: A list of variables that need to be printed.
    """
    result = ""
    variable_list = sorted(variable_list, key=lambda res: res.name, reverse=True)  # This makes sure nice printing
    for variable in variable_list:
        result += f"{variable.name}={variable.value} "
    print(result)
    pass


def plotDiagram(data, results, ylabel):
    """
    Function that encapsulates the creation of a plot that showcases the query results. It plots a time-series line,
    points for the queried x's and boxes for the queried offsets.

    :param data: The data used for the line plot.
    :param results: The results used for the points and boxes.
    :param ylabel: The name of the y axis
    :return: The fig object of the plot.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    x, y = data[:, 0], data[:, 1]

    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title("Query Results", weight='bold')
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    # Line plot
    ax.plot(x, y, color='#0868AC')

    # Splitting result values in different lists
    tiks = []
    offsets = []
    for result in results:
        tik = getTimeIdIndex_Binary(result)

        tiks.append(result[tik].value)
        offsets.append(result[not tik].value)

    # For each x value that is in the result set, we find the indexes of the relevant results and a point and boxes.
    for x1, y1 in zip(x, y):
        if x1 in tiks:
            indexes = [i for (i, e) in enumerate(tiks) if e == x1]
        else:
            continue

        # Point draw
        ax.scatter(x1, y1, c="#5AAA95", zorder=4)

        # Boxes for all corresponding offset values found.
        height = (max(y) - min(y)) * 0.2
        for index in indexes:
            width = offsets[index]
            rect = patches.Rectangle((x1, y1 - height / 2), width, height, zorder=3, linewidth=2, edgecolor='#FF9F1C',
                                     facecolor='none')
            ax.add_patch(rect)

    return fig


####################################################################################################################
#                                                Constraint UDFs
####################################################################################################################


def avg_amp(data_chunk):
    """
    Simply gets a chunk of data, calculates the Average Signal Amplitude and returns it.
    """
    return np.average(data_chunk)


def max_amp_excess_right(data, x, lx, args):
    """
    Maximum Signal Amplitude (MSA) is calculated in the interval, and in the right neighborhood that is determined from
    the argument. The difference is returned.

    :param data: The time-series table
    :param x: The starting point.
    :param lx: The interval size
    :param args: List with arguments for this UDF. Contains only the neighborhood size.
    :return: The difference between the MSAs of the two regions.
    """
    neighborhood_size = args[0]
    x_prime = x + lx
    max_time = max(data[:, 0])
    if x_prime + neighborhood_size > max_time:
        lx_prime = max_time - x_prime
    else:
        lx_prime = neighborhood_size

    start = int(np.where(data[:, 0] == x)[0])
    end = int(np.where(data[:, 0] == x + lx)[0]) + 1

    start_prime = int(np.where(data[:, 0] == x_prime)[0])
    end_prime = int(np.where(data[:, 0] == x_prime + lx_prime)[0]) + 1

    return max(data[start:end, 1]) - max(data[start_prime:end_prime, 1])


def max_amp_excess_left(data, x, lx, args):
    """
    Maximum Signal Amplitude (MSA) is calculated in the interval, and in the left neighborhood that is determined from
    the argument. The difference is returned.

    :param data: The time-series table
    :param x: The starting point.
    :param lx: The interval size
    :param args: List with arguments for this UDF. Contains only the neighborhood size.
    :return: The difference between the MSAs of the two regions.
    """
    neighborhood_size = args[0]
    if x - neighborhood_size < min(data[:, 0]):
        neighborhood_size = x - min(data[:, 0])

    start = int(np.where(data[:, 0] == x)[0])
    end = int(np.where(data[:, 0] == x + lx)[0]) + 1

    start_prime = int(np.where(data[:, 0] == x - neighborhood_size)[0])
    end_prime = int(np.where(data[:, 0] == x)[0]) + 1

    return max(data[start:end, 1]) - max(data[start_prime:end_prime, 1])
