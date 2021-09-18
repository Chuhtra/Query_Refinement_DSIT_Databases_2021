import copy
import random as r


class Failpoint:
    """
    Class utilized to represent nodes of the search tree that should be stored as fail points to be run later.
    """

    def __init__(self, node, maximal_bounds, failed_constraints, VCs):
        """
        Constructor for Failpoint Class

        :param node: The node that failed the constraint checks.
        :param maximal_bounds: The recorded a' and b' domain boundaries, for which all leaves under this node would
        succeed the constraint test. (Definition from the paper)
        :param failed_constraints: A list containing the names of the constraints that failed for this node.
        :param VCs: A dictionary with the ratio of failed constraints for the node's leaves. Useful for PR calculation.
        """
        self.node = node
        self.maximal_bounds = maximal_bounds
        self.failed_constraints = failed_constraints
        self.VCs = VCs


class NodeVariable:
    """
    Class representing a tree node variable.
    """

    def __init__(self, name, var_min, var_max, target=None):
        """
        Constructor for NodeVariable class. This class is used both for domain and for constraint variables.

        :param name: The name of the recorded variable.
        :param var_min: The minimum domain boundary for the variable.
        :param var_max: The maximum domain boundary for the variable.
        :param target: Useful only for constraint variables and more specifically for query tightening. It's either
        maximization or minimization.
        """
        self.name = name
        self.min = var_min
        self.max = var_max
        if target is None:
            self.value = None  # Used when the NodeVariable is bounded, instead of min/max. For code readability.
        else:
            self.target = target

        if self.min == self.max:
            self.value = self.max


class TreeNode:
    """
    Class representing a search tree node.
    """

    def __init__(self, decision_variables=None, domains=None):
        """
        Constructor of the tree node class.

        :param decision_variables: List with the wanted decision variable names.
        :param domains: List with the wanted decision variable domains.
        """
        self.visited = False  # Flag showing whether the node was visited during search.
        self.children = []  # List that will hold the two children for non-leaf nodes.
        self.parent = None  # Pointer to the parent node.
        self.variables = []  # List containing the variables that are BOUNDED
        self.unbound_variables = []  # List containing the variables that are UNBOUNDED
        self.BRP = None  # Best Relaxation Penalty (used for failpoints in query relaxation)
        self.WRP = None  # Worst Relaxation Penalty (used for failpoints in query relaxation)

        if decision_variables is not None and domains is not None:
            for i in range(len(decision_variables)):
                if domains[i][0] != domains[i][1]:  # This handles if a domain is bounded right from the start.
                    self.unbound_variables.append(NodeVariable(decision_variables[i], domains[i][0], domains[i][1]))
                else:
                    self.variables.append(NodeVariable(decision_variables[i], domains[i][0], domains[i][1]))

    def addChild(self, child):
        """
        Simple method that establishes parent-child connection between two nodes.

        :param child: The node that should be connected under the current node.
        """
        self.children.append(child)


class Tree:
    """
    Class representing the binary search tree (BST).
    """

    def __init__(self, node, cp_constraints, constraint_limits=None, constraint_targets=None):
        """
        Constructor of the BST class.

        :param node: The node that is about to be the tree root.
        :param cp_constraints: The constraints that define the query that is searched in this tree.
        :param constraint_limits: The domain boundaries of the provided constraints.
        :param constraint_targets: The maximization/minimization targets for each provided constraint.
        """
        self.root = node
        self.root.visited = False  # Stated here cause the used node may have been visited in some previous tree.
        self.current_node = self.root  # Assignment of the search pointer.
        self.MRK = None  # Minimum Rank, metric utilized for tightening and determined by the "worse" result in the set.
        self.MRP = 1  # Maximum Relaxation Penalty, metric used for subtree pruning.

        self.constraints = []
        if isinstance(cp_constraints[0], str):  # Useful if, cause constraints of relaxation are already NodeVariables.
            for i in range(len(cp_constraints)):
                self.constraints.append(NodeVariable(cp_constraints[i], constraint_limits[i][0],
                                                     constraint_limits[i][1], constraint_targets[i]))
        elif isinstance(cp_constraints[0], NodeVariable):
            self.constraints = copy.deepcopy(cp_constraints)

    def backtrack(self):
        """
        Simple function that paints current node as visited before traversing back to find the next non-visited node.
        This node can be a sibling one, a predecessor or a distant one. It simply repositions the search pointer.
        """
        self.current_node.visited = True

        while self.current_node.visited:  # Continue until a non-visited node is found.
            if self.current_node == self.root:  # If pointer has returned to the root, then tree is completely searched.
                return

            for next_child in self.current_node.parent.children:  # For backtracking, first check the sibling.
                if not next_child.visited:
                    self.current_node = next_child
                    return

            self.current_node = self.current_node.parent  # If both siblings are visited then move a step up.

    def branchOut(self):
        """
        Function that finds a branching point to create two children for a non-leaf node. Children and parents are
        identical in all aspects except some decision variables.

        :return: The two created children nodes.
        """
        # Initializations
        child1 = TreeNode()
        child2 = TreeNode()

        # Random selection of an unbound variable
        rand_index = r.randint(0, len(self.current_node.unbound_variables) - 1)
        var_max = self.current_node.unbound_variables[rand_index].max
        var_min = self.current_node.unbound_variables[rand_index].min

        # New branching point (arbitrary method of selection)
        if var_max is not None and var_min is not None:  # Ideally we break at the intersection.
            breaking = var_min + ((var_max - var_min) // 2)
        elif var_max is not None:  # If left-unbounded we divide max by 2.
            breaking = var_max // 2
        elif var_min is not None:  # If right-unbounded we choose the square as a breaking point.
            breaking = var_min ** 2
        else:  # If variable is completely unbounded then we get a random point.
            import numpy as np
            breaking = rand_index = r.randint(0, np.inf)

        # Connect children to the parent
        for child in [child1, child2]:
            self.current_node.addChild(child)
            child.parent = self.current_node

            # Copy variables of the parent to the children.
            child.unbound_variables = copy.deepcopy(self.current_node.unbound_variables)
            child.variables = copy.deepcopy(self.current_node.variables)

        # Use variable branching point on children (edit variable domains)
        child1.unbound_variables[rand_index].max = breaking
        child2.unbound_variables[rand_index].min = breaking + 1 if breaking + 1 <= child2.unbound_variables[
            rand_index].max else child2.unbound_variables[rand_index].max

        # Variable bounding check of the changed variable domain for both children
        for child in [child1, child2]:
            if child.unbound_variables[rand_index].min == child.unbound_variables[rand_index].max:
                child.unbound_variables[rand_index].value = child.unbound_variables[rand_index].min

                del child.unbound_variables[rand_index].min
                del child.unbound_variables[rand_index].max

                child.variables.append(child.unbound_variables[rand_index])
                child.unbound_variables.remove(child.unbound_variables[rand_index])

        return child1, child2
