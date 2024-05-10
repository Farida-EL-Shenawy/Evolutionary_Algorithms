import collections
import random
import matplotlib.pyplot as plt

__all__ = ['DE']
Individual = collections.namedtuple('Individual', 'ind fit')


class DE(object):
    """This class implements differential evolution."""

    def __init__(self, x='rand', y=1, z='bin', F=.5, CR=.1):
        self.x = x
        self.y = y
        self.z = z
        self.F = F
        self.CR = CR

    def solve(self, fitness, initial_population, iterations=1000):
        current_generation = [Individual(ind, fitness(*ind)) for ind in initial_population]
        best_fitnesses = []  # To store the best fitness values over iterations

        for _ in range(iterations):
            trial_generation = []

            for ind in current_generation:
                v = self._mutate(current_generation)
                u = self._crossover(ind.ind, v)
                trial_generation.append(Individual(u, fitness(*u)))

            current_generation = self._selection(current_generation, trial_generation)

            # Track the best fitness value
            best_fitness = min(ind.fit for ind in current_generation)
            best_fitnesses.append(best_fitness)

        best_index = self._get_best_index(current_generation)
        best_solution = current_generation[best_index].ind
        return best_solution, best_fitnesses

    def _mutate(self, population):
        if self.x == 'rand':
            r1, *r = self._get_indices(self.y * 2 + 1, len(population))
        elif self.x == 'best':
            r1 = self._get_best_index(population)
            r = self._get_indices(self.y * 2, len(population), but=r1)

        mutated = population[r1].ind[:]  # copy base vector
        dimension = len(mutated)
        difference = [0] * dimension

        for plus in r[:self.y]:
            for i in range(dimension):
                difference[i] += population[plus].ind[i]

        for minus in r[self.y:]:
            for i in range(dimension):
                difference[i] -= population[minus].ind[i]

        for i in range(dimension):
            mutated[i] += self.F * difference[i]

        return mutated

    def _crossover(self, x, v):
        u = x[:]
        i = random.randrange(len(x))  # NP

        for j, (a, b) in enumerate(zip(x, v)):
            if i == j or random.random() <= self.CR:
                u[j] = v[j]

        return u

    def _selection(self, current_generation, trial_generation):
        generation = []

        for a, b in zip(current_generation, trial_generation):
            if a.fit < b.fit:
                generation.append(a)
            else:
                generation.append(b)

        return generation

    def _get_indices(self, n, upto, but=None):
        candidates = list(range(upto))

        if but is not None:
            candidates.remove(but)

        return random.sample(candidates, n)

    def _get_best_index(self, population):
        min_fitness = population[0].fit
        best = 0

        for i, x in enumerate(population):
            if x.fit < min_fitness:
                best = i

        return best


if __name__ == '__main__':
    import math

    def sum_of_squares(*args):
        return sum(x ** 2 for x in args)

    # Input parameters from the user
    x_strategy = input("Enter mutation strategy (rand/best): ")
    y_vectors = int(input("Enter the number of difference vectors: "))
    z_scheme = input("Enter crossover scheme (bin): ")
    F_weight = float(input("Enter mutation weight (F): "))
    CR_probability = float(input("Enter crossover probability (CR): "))
    population_size = int(input("Enter population size: "))
    max_generations = int(input("Enter maximum generations: "))

    de = DE(x=x_strategy, y=y_vectors, z=z_scheme, F=F_weight, CR=CR_probability)
    bound = 10  # Adjust bound based on your requirement
    pop = [[random.uniform(-bound, bound) for _ in range(2)] for _ in range(population_size)]

    best_solution, best_fitnesses = de.solve(sum_of_squares, pop, iterations=max_generations)

    print("Best solution found:", best_solution)
    print("Best fitness:", sum_of_squares(*best_solution))

    # Plot the best fitness over iterations
    plt.plot(range(1, max_generations + 1), best_fitnesses)
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness over Iterations")
    plt.show()
