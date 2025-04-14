class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        """
        Gradient Descent for X^2.
        """
        if init == 0:
            return float(init)
        else:
            for i in range(iterations):
                gradient = 2*init
                init = init - gradient * learning_rate
            final_result = round(init,5)
            return final_result

#Test
Solution = Solution()
iterations = 50
learning_rate = 0.05
init = 2
result = Solution.get_minimizer(iterations=iterations, learning_rate=learning_rate, init=init)
print(result)