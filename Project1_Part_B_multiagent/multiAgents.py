# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        v = -99999999
        rstAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.value(nextState, 1, 0)
            if nextValue > v:
                v = nextValue
                rxtAction = action
        return rxtAction
        util.raiseNotDefined()
    def isTerminal(self, gameState, currentDepth):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return True
        else:
            return False

    def value(self, gameState, currentAgent, currentDepth):
        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState) # should be a value
        if currentAgent == 0:
            return self.maxValue(gameState, currentAgent, currentDepth)
        else:
            return self.minValue(gameState, currentAgent, currentDepth)
    
    def maxValue(self, gameState, currentAgent, currentDepth):
        v = -99999999
        actions = gameState.getLegalActions(currentAgent)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            v = max(v, self.value(successor, 1, currentDepth))
        return v

    def minValue(self, gameState, currentAgent, currentDepth):
        v = 99999999
        actions = gameState.getLegalActions(currentAgent)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            if currentAgent == (gameState.getNumAgents() - 1):
                v = min(v, self.value(successor, 0, currentDepth+1))
            else:
                v = min(v, self.value(successor, currentAgent+1, currentDepth))
        return v 






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = -99999999
        beta = 99999999
        v = -99999999
        rstAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.value(nextState, 1, 0, alpha, beta)
            if nextValue > v:
                v = nextValue
                rxtAction = action
            alpha = max(alpha, v)
        return rxtAction
        util.raiseNotDefined()

    def isTerminal(self, gameState, currentDepth):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return True
        else:
            return False
    
    def value(self, gameState, currentAgent, currentDepth, alpha, beta):
        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState) # should be a value
        if currentAgent == 0:
            return self.maxValue(gameState, currentAgent, currentDepth, alpha, beta)
        else:
            return self.minValue(gameState, currentAgent, currentDepth, alpha, beta)

    def maxValue(self, gameState, currentAgent, currentDepth, alpha, beta):
        v = -99999999
        for action in gameState.getLegalActions(currentAgent):
            successor = gameState.generateSuccessor(currentAgent, action)
            v = max(v, self.value(successor, 1, currentDepth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, currentAgent, currentDepth, alpha, beta):
        v = 99999999
        actions = gameState.getLegalActions(currentAgent)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            if currentAgent == (gameState.getNumAgents() - 1):
                v = min(v, self.value(successor, 0, currentDepth+1, alpha, beta))
            else:
                v = min(v, self.value(successor, currentAgent+1, currentDepth, alpha, beta))
            if v < alpha:
                return v
            beta = min(v, beta)
        return v 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        v = -99999999
        rstAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.value(nextState, 1, 0)
            if nextValue > v:
                v = nextValue
                rxtAction = action
        return rxtAction

    def isTerminal(self, gameState, currentDepth):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return True
        else:
            return False
    
    def value(self, gameState, currentAgent, currentDepth):
        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState) # should be a value
        if currentAgent == 0:
            return self.maxValue(gameState, currentAgent, currentDepth)
        else:
            return self.expValue(gameState, currentAgent, currentDepth)

    def maxValue(self, gameState, currentAgent, currentDepth):
        v = -99999999
        actions = gameState.getLegalActions(currentAgent)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            v = max(v, self.value(successor, 1, currentDepth))
        return v
    
    def expValue(self, gameState, currentAgent, currentDepth):
        v = 0.0
        actions = gameState.getLegalActions(currentAgent)
        for action in actions:
            successor = gameState.generateSuccessor(currentAgent, action)
            p = 1.0 / len(actions)
            if currentAgent == (gameState.getNumAgents() - 1):
                v = v + self.value(successor, 0, currentDepth+1)
            else:
                v = v + self.value(successor, currentAgent+1, currentDepth)
        return v 

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

    stateScore = currentGameState.getScore()
    finalScore = stateScore
    newFoodList = newFood.asList()
    scaryGhosts = []
    # eat capsule!
    sum_capsules = sum(manhattanDistance(newPos, food) for food in newCapsules)
    finalScore -=  2*sum_capsules
    # get away from ghosts!
    for i in range(len(newGhostStates)):
        if newScaredTimes[i] > 0:
            newFoodList.append(newGhostStates[i].getPosition())
        else:
            scaryGhosts.append(newGhostStates[i])
    sum_scary = sum(manhattanDistance(newPos, ghost.getPosition()) for ghost in scaryGhosts)
    finalScore +=  2*sum_scary
    # get food!
    heuristic = sorted(manhattanDistance(newPos, food) for food in newFoodList)
    sum_heuristic = sum(heuristic[:3]) + 5 * currentGameState.getNumFood()
    finalScore -= sum_heuristic


        
    return finalScore

# Abbreviation
better = betterEvaluationFunction
