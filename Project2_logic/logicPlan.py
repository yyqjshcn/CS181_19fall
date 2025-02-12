# logicPlan.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr("A")
    B = logic.Expr("B")
    C = logic.Expr("C")
    s1 = A | B
    s2 = (~A) % ((~B) | C)
    s3 = logic.disjoin([(~A), (~B) , C])
    return logic.conjoin([s1, s2, s3])

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr("A")
    B = logic.Expr("B")
    C = logic.Expr("C")
    D = logic.Expr("D")
    s1 = C % (B | D)
    s2 = A >> ((~B) & (~D))
    s3 = (~(B&(~C))) >> A
    s4 = (~D) >> C
    return logic.conjoin([s1, s2, s3, s4])

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    A = logic.PropSymbolExpr("WumpusAlive[0]")
    B = logic.PropSymbolExpr("WumpusAlive[1]")
    C = logic.PropSymbolExpr("WumpusBorn[0]")
    D = logic.PropSymbolExpr("WumpusKilled[0]")
    s1 = B % ((A & (~D)) | ((~A) & C))
    s2 = ~(A & C)
    s3 = C
    return logic.conjoin([s1, s2, s3])

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    return logic.pycoSAT(logic.to_cnf(sentence))

def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    return logic.disjoin(literals)


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    result = []
    for i in literals:    # Suppose i to be true
        for j in literals:
            if i != j:
                result.append((~i|~j))
            else:
                continue        
    return logic.conjoin(result)
    


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin([atMostOne(literals), atLeastOne(literals)])


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    key = list(model.keys())
    val = list(model.values())
    result = []
    for i in range(len(val)):
        if val[i] == True:
            operation = logic.PropSymbolExpr.parseExpr(key[i])
            if operation[0] in actions:
                result.append(operation)
            
    def takeSecond(elem):
        return elem[1]
    for i in range(len(result)): 
        result[i] = [result[i][0], int(result[i][1])]
    result.sort(key=takeSecond)
    return ([i[0] for i in result])


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    walls=walls_grid.asList()
    actions = ['North', 'South', 'East', 'West']
    operation = []
    for action in actions:
        if action == 'North':
            if (x, y-1) not in walls:
                operation.append(logic.conjoin(logic.PropSymbolExpr(pacman_str, x, y-1, t-1), logic.PropSymbolExpr('North',t-1)))
        if action == 'South':
            if (x, y+1) not in walls:
                operation.append(logic.conjoin(logic.PropSymbolExpr(pacman_str, x, y+1, t-1), logic.PropSymbolExpr('South',t-1)))
        if action == 'East':
            if (x-1, y) not in walls:
                operation.append(logic.conjoin(logic.PropSymbolExpr(pacman_str, x-1, y, t-1), logic.PropSymbolExpr('East',t-1)))
        if action == 'West':
            if (x+1, y) not in walls:
                operation.append(logic.conjoin(logic.PropSymbolExpr(pacman_str, x+1, y, t-1), logic.PropSymbolExpr('West',t-1)))
    return logic.PropSymbolExpr(pacman_str, x, y, t) % logic.disjoin(operation)
    # return logic.Expr('A') # Replace this with your expression


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    
    "*** YOUR CODE HERE ***"
    actions = ['North', 'South', 'East', 'West']
    startState = []
    initialX, initialY = problem.getStartState()
    initial = logic.PropSymbolExpr(pacman_str, initialX, initialY, 0)
    goalX, goalY = problem.getGoalState()
    goalSuccessors = []
    goalActions = []
    
    steps = 50
    for x in range(1, width+1):
        for y in range(1, height+1):
            if not (x, y) in walls.asList():
                startState.append(logic.PropSymbolExpr(pacman_str, x, y, 0))
    for i in range(1, steps):
        successors = []
        start = exactlyOne(startState)  # needed for goal
        goal = logic.PropSymbolExpr(pacman_str, goalX, goalY, i)
        for x in range(1, width+1):
            for y in range(1, height+1):
                if (x,y) not in walls.asList():
                    successors.append(pacmanSuccessorStateAxioms(x, y, i, walls))
        successor = logic.conjoin(successors)
        action = []
        for a in actions:
            action.append(logic.PropSymbolExpr(a, i-1))
        goalActions.append(exactlyOne(action))
        goalSuccessors.append(successor)
        # print("start: ", start)
        # print("initial: ", initial)
        # print("goal: ", goal)
        # print("goalActions: ", goalActions)
        # print("goalSuccessors: ", goalSuccessors)
        isGoal = findModel(logic.conjoin([start, initial, goal, logic.conjoin(goalSuccessors), logic.conjoin(goalActions)]))
        if isGoal:
            return extractActionSequence(isGoal, actions) 


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    actions = ('North', 'South', 'East', 'West')
    startState = []
    startX, startY = problem.getStartState()[0]        # Get the start state axis
    foodTemp = problem.getStartState()[1]              # Get all the food
    food = foodTemp.asList()
    initial = logic.PropSymbolExpr(pacman_str, startX, startY, 0) # needed for goal
    goalSuccessors = []
    goalActions = []
    steps = 50
    for x in range(1, width+1):
        for y in range(1, height+1):
            if not (x, y) in walls.asList():
                if (x, y) == (startX, startY):
                    startState.append(logic.PropSymbolExpr(pacman_str, x, y, 0))
                else:
                    startState.append(~logic.PropSymbolExpr(pacman_str, x, y, 0))
    
    for i in range(1, steps+1):
        foodLeft = []
        successors = []
        start = logic.conjoin(startState)
        # Deal with food
        for f in food:
            foodList = []
            for j in range(i):
                foodList.append(logic.PropSymbolExpr(pacman_str,f[0],f[1],j))            
            foodLeft.append(atLeastOne(foodList))

        # Deal with successors
        for x in range(1, width+1):
            for y in range(1, height+1):
                if (x,y) not in walls.asList():
                    successors.append(pacmanSuccessorStateAxioms(x, y, i, walls))

        successor = logic.conjoin(successors)
        action = []
        for a in actions:
            action.append(logic.PropSymbolExpr(a, i-1))
        goalActions.append(exactlyOne(action))
        goalSuccessors.append(successor)
        # print("start: ", start)
        # print("initial: ", initial)
        # print("food: ", foodLeft)
        # print("goalActions: ", goalActions)
        # print("goalSuccessors: ", goalSuccessors)
        isGoal = findModel(logic.conjoin([start, initial, logic.conjoin(foodLeft), logic.conjoin(goalSuccessors), logic.conjoin(goalActions)]))
        if isGoal:
            return extractActionSequence(isGoal, actions) 

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    