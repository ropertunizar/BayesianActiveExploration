class RewardFunction:
    def __call__(self, state, action, next_state):
        """ task reward of a single real transition (numpy arrays) """
        raise NotImplementedError


class Task:
    def __init__(self, measure, reward_function):
        """
        Args:
            measure: batched reward used to learn the task policy inside the model (a utilities.Measure)
            reward_function: reward used to score the policy in the real environment
        """
        self.measure = measure
        self.reward_function = reward_function
