import numpy as np
from itertools import product


class TwoPlayerBlottoGame:

    def __init__(self, soldiers, battlefields=2):
        self.soldiers = soldiers
        self.battlefields = battlefields
        self.actions = generate_combinations(soldiers, battlefields)
        self.numofstrats = len(self.actions)
        # First convention used
        self.pmat = np.zeros((self.numofstrats, self.numofstrats))
        for row in range(self.numofstrats):
            for column in range(self.numofstrats):
                self.pmat[row, column] += payoff_calculator(self.actions[row], self.actions[column])[0]

    def learn(self, player1, player2, runlength):
        player1.iterations = 0
        player2.iterations = 0
        player1.playinit(self.numofstrats, self.pmat)
        player2.playinit(self.numofstrats, self.pmat)
        # Add to basic agent class init
        for game in range(runlength):
            action1 = player1.chooseaction()
            action2 = player2.chooseaction()
            payoff = (self.pmat[action1, action2], self.pmat.T[action1, action2])
            player1.iterations += 1
            player2.iterations += 1
            player1.update(payoff[0], action1, action2)
            player2.update(payoff[1], action2, action1)
        player1.actions = self.actions
        player2.actions = self.actions
        player1.outcome()
        player2.outcome()  # To hstack with results at the end for a lovely table


class Agent:
    """Basic Agent class."""

    def __init__(self):
        self.iterations = 0
        self.result = None


class Sampleaverage(Agent):
    """Agent employing sample-average update method."""

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def playinit(self, options, pmat):
        self.timesselected = np.zeros(options)
        self.sample_average = np.zeros(options)
        self.results = []
        self.options = options

    def chooseaction(self):
        if self.epsilon > np.random.uniform(0, 1):
            i = np.random.randint(0, high=self.options)
        else:
            i = np.argmax(self.sample_average)
        return i

    def update(self, payoff, i, oppaction):
        self.results.append(payoff)
        self.timesselected[i] += 1
        self.sample_average[i] = rewardsestimate(self.sample_average[i], payoff, self.timesselected[i])

    def outcome(self):
        self.sample_average = self.sample_average[:, np. newaxis]
        self.exactresult = np.hstack((np.array(self.actions), self.sample_average))
        self.result = np.round(self.exactresult, decimals=2)


class RegretMatching(Agent):

    def __init__(self):
        super().__init__()

    def playinit(self, options, pmat):
        self.cum_regrets = np.zeros((options, 1))
        self.stratprobsum = np.zeros((options, 1))
        self.options = options
        self.pmat = pmat
        self.convergence = []

    def chooseaction(self):
        # Select Strategy profile for p1
        if np.all(self.cum_regrets[:, 0] <= 0):
            self.stratp1 = [1 / self.options] * self.options
        else:
            posregrets1 = []
            for regret in self.cum_regrets[:, 0]:
                if regret > 0:
                    posregrets1.append(regret)
                else:
                    posregrets1.append(0)
            normaliser1 = sum(posregrets1)
            self.stratp1 = [x / normaliser1 for x in posregrets1]
        return probvectoractionselector(self.stratp1)

    def update(self, payoff, i, oppaction):
        # Add strategy profile to strategy profile sum
        self.stratprobsum[:, 0] += self.stratp1
        # p1best = self.actions[np.argmax(self.stratprobsum[:, 0])]
        # Calculate regrets
        p1regrets = (self.pmat[:, oppaction] - [payoff] * self.options)
        # Add regrets to cumulative regrets
        self.cum_regrets[:, 0] += p1regrets
        # print(self.stratprobsum.shape)
        # print(self.actions.shape)
        self.convergence.append(1 - (sum(self.stratprobsum[[2, 3, 7, 9, 11, 14, 15, 16, 17]])) / self.iterations)

    def outcome(self):
        scores = np.hstack((np.array(self.actions),
                            self.stratprobsum / (self.iterations * np.ones(self.stratprobsum.shape))))
        self.result = scores[scores[:, -1] > 0.001]


class ReinforementLearner(Agent):
    """Add choice=  for action choice method and model= to choose model.

    Choices:
    eps = Epsilon greedy
    soft = Softmax
    none = Proportional

    Models:
    qlearn = Single state Q-learning (specify alpha)
    cross = Cross (specify eta)
    erevroth = Erev-Roth Cumulative payoff matching (specify eta)
    arthur = Arthur
    """

    def __init__(self, choice=None, epsilon=0.9, model=None, eta=0.5, alpha=0.6):
        super().__init__()
        self.choice = choice
        self.epsilon = epsilon
        self.model = model
        self.eta = eta
        self.alpha = alpha

    def playinit(self, options, pmat):
        self.propensity = np.zeros(options)
        self.pmat = pmat
        self.options = options

    def chooseaction(self):
        if self.choice == "eps":
            return epsilon(self.propensity, self.epsilon, self.options)
        elif self.choice == "soft":
            return softmax(self.propensity)
        else:
            return probvectoractionselector(self.propensity)

    def update(self, payoff, i, oppaction):
        if self.model == "arthur":
            return arthur(self.propensity, payoff, i, self.iterations)
        if self.model == "cross":
            return cross(self.propensity, payoff, i, self.eta)
        if self.model == "erevroth":
            return erevroth(self.propensity, payoff, i, self.eta)
        if self.model == "qlearn":
            return qlearn(self.propensity, payoff, i, self.alpha)

    def outcome(self):
        # print(np.array(self.actions).shape)
        # print(np.ones(np.array(self.propensity).shape))
        scores = np.hstack((np.array(self.actions),
                            np.array(self.propensity)[:, None]))
        # print(scores[:, -1])
        scores[:, -1] = scores[:, -1] / sum(scores[:, -1])
        self.result = scores[scores[:, -1] > 0.001]


def payoff_calculator(dist1, dist2):
    p1util = 0
    p2util = 0
    for num in range(len(dist1)):
        if dist1[num] > dist2[num]:
            p1util += 1
        elif dist1[num] < dist2[num]:
            p2util += 1
    if p1util > p2util:
        p1payoff = 1
        p2payoff = -1
    elif p1util == p2util:
        p1payoff = 0
        p2payoff = 0
    else:
        p1payoff = -1
        p2payoff = 1
    # Test to see what would happen if we changed payoffs
    # to number of battlefields won instead
    # p1altpayoff = p1util - p2util
    # p2altpayoff = p2util - p1util
    return [p1payoff, p2payoff]


def generate_combinations(s, n):
    """Generate all combinations of s soldiers over n battlefields."""
    combinations = product(range(s + 1), repeat=n)

    # Filter out combinations where the sum exceeds s
    result = [combination for combination in combinations if sum(combination) == s]

    return result


def probvectoractionselector(vector):
    """Chooses an index based on a vector of probabilities.

    We choose an index based on proportionality
    We iterate through the vector, summing till a random number is reached.
    """
    # assert math.isclose(sum(vector), 1, rel_tol=1e-5, abs_tol=0.0)
    if sum(vector) <= 0:
        normalised = (1 / len(vector)) * np.ones(len(vector))
    else:
        totalsum = sum(vector)
        normalised = np.array(vector) / totalsum

    # Generate a random number between 0 and 1
    rand_num = np.random.uniform()

    # Accumulate probabilities until the random number is reached
    cumulative_probability = 0
    for index, prob in enumerate(normalised):
        cumulative_probability += prob
        if rand_num <= cumulative_probability:
            return index


def rewardsestimate(qn, rn, n):
    qn += (rn - qn) / n
    return qn


def epsilon(vector, epsilon, options):
    if epsilon > np.random.uniform(0, 1):
        i = np.random.randint(0, high=options)
    else:
        i = np.argmax(vector)
    return i


def softmax(vec):
    newvec = np.exp(vec)
    return probvectoractionselector(newvec)


def cross(vector, payoff, i, eta):
    vector -= eta * payoff * vector
    vector[i] += eta * payoff


def erevroth(vector, payoff, i, eta):
    vector[i] += eta * payoff


def arthur(vector, payoff, i, iteration, constant=0.5):
    if iteration == 1:
        vector = np.ones(len(vector)) * (constant / len(vector))
    else:
        vector[i] += payoff + 1
        vector = vector * ((constant * (iteration + 1)) / ((constant * iteration) + payoff + 1))


def qlearn(propensity, payoff, i, alpha):
    x = propensity[i]
    propensity[i] += (alpha * (payoff + 1 - x))
