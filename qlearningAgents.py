# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.Q = {}


        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        Q = self.Q

        # if (state, action) in Q, return value, otherwise return 0
        if ((state, action) in Q):
            return Q[(state, action)]
        else:
            return 0

    # compute both max value and corresponding action from Q-values
    def computeValueActionFromQValues(self, state):
        # get all legal actions
        actions = self.getLegalActions(state)
        Q = self.Q

        # if no actions, return 0 and None
        if len(actions) == 0:
            return 0, None

        # find action which maximizes Q-value for given state
        L = [ (self.getQValue(state, action), action) for action in actions]
        val, argMax = max(L)
        # return both value and action
        return val, argMax


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        # value part of computeValueActionFromQValues
        val, argMax = self.computeValueActionFromQValues(state)
        return val

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        # action part of computeValueActionFromQValues
        val, argMax = self.computeValueActionFromQValues(state)
        return argMax


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action


        legalActions = self.getLegalActions(state)
        # flip the coin with probablity self.epsilon
        if util.flipCoin(self.epsilon):
            # if head, return random action
            return random.choice(legalActions)
        else:
            # otherwise return one maximizing Q value
            return self.computeActionFromQValues(state)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # learning rate and discount
        alpha = self.alpha
        gamma = self.discount

        # old Q(state, action)
        qOld = self.getQValue(state, action)
        # updated value
        qNew = (1-alpha) * qOld + alpha * (reward + gamma * self.getValue(nextState))

        self.Q[(state, action)] = qNew

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        s = 0
        featureVector = self.featExtractor.getFeatures(state, action)

        w = self.getWeights()
        for feat in featureVector:
            # for each feature in vector, if weight is non-zero,
            # include it in the sum
            if feat in w:
                s += w[feat] * featureVector[feat]

        # return the sum
        return s

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        # maximum of Q values over all possible actions from nextAction
        # or zero if there are no further actions
        L  = [self.getQValue(nextState, act) for act in self.getLegalActions(nextState)]
        maxQ = max(L) if len(L) > 0 else 0

        # difference
        dif = reward + self.discount * maxQ - self.getQValue(state, action)

        # get weights and feature vectors
        w = self.getWeights()
        featureVector = self.featExtractor.getFeatures(state, action)

        # weights update
        for feat in featureVector:
            # if it is non-null, update the value
            if feat in w:
                w[feat] += self.alpha * dif * featureVector[feat]
            else:
                # otherwise just put it in dict
                w[feat] = self.alpha * dif * featureVector[feat]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
