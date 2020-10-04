class ProblemSetup:

    def __init__(self):
        self.prob_matrix = [[[0.3, 0.4, 0.3], [0.1, 0.2, 0.7]],
                                   [[0.2, 0.2, 0.6], [0.5, 0.5, 0]],
                                   [[0.3, 0.1, 0.6], [0.3, 0.4, 0.3]]]

        self.reward_matrix = [[[2, 5, 3], [4, 2, 8]],
                              [[12, 15, 14], [8, -3, 3]],
                              [[2, 7, -4], [10, 18, 14]]]

        self.time_epochs = range(10)
        self.actions = range(2)
        self.states = range(3)
