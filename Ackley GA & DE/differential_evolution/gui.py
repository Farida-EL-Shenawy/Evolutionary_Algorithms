import tkinter as tk
from tkinter import ttk
import random
import matplotlib.pyplot as plt
import math
import collections
import io
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


class DifferentialEvolutionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Differential Evolution for Ackley Function")

        self.create_widgets()

    def create_widgets(self):
        # Mutation strategy
        self.strategy_label = ttk.Label(self.root, text="Mutation Strategy:")
        self.strategy_label.grid(row=0, column=0, padx=5, pady=5)
        self.strategy_var = tk.StringVar(self.root)
        self.strategy_var.set("rand")  # Default value
        self.strategy_menu = ttk.OptionMenu(
            self.root, self.strategy_var, "rand", "rand", "best"
        )
        self.strategy_menu.grid(row=0, column=1, padx=5, pady=5)

        # Number of difference vectors
        self.vectors_label = ttk.Label(self.root, text="Number of Vectors:")
        self.vectors_label.grid(row=1, column=0, padx=5, pady=5)
        self.vectors_entry = ttk.Entry(self.root)
        self.vectors_entry.grid(row=1, column=1, padx=5, pady=5)

        # Crossover scheme
        self.crossover_label = ttk.Label(self.root, text="Crossover Scheme:")
        self.crossover_label.grid(row=2, column=0, padx=5, pady=5)
        self.crossover_var = tk.StringVar(self.root)
        self.crossover_var.set("bin")  # Default value
        self.crossover_menu = ttk.OptionMenu(
            self.root, self.crossover_var, "bin"
        )
        self.crossover_menu.grid(row=2, column=1, padx=5, pady=5)

        # Mutation weight
        self.F_label = ttk.Label(self.root, text="Mutation Weight (F):")
        self.F_label.grid(row=3, column=0, padx=5, pady=5)
        self.F_entry = ttk.Entry(self.root)
        self.F_entry.grid(row=3, column=1, padx=5, pady=5)

        # Crossover probability
        self.CR_label = ttk.Label(self.root, text="Crossover Probability (CR):")
        self.CR_label.grid(row=4, column=0, padx=5, pady=5)
        self.CR_entry = ttk.Entry(self.root)
        self.CR_entry.grid(row=4, column=1, padx=5, pady=5)

        # Population size
        self.population_label = ttk.Label(self.root, text="Population Size:")
        self.population_label.grid(row=5, column=0, padx=5, pady=5)
        self.population_entry = ttk.Entry(self.root)
        self.population_entry.grid(row=5, column=1, padx=5, pady=5)

        # Maximum generations
        self.generations_label = ttk.Label(self.root, text="Maximum Generations:")
        self.generations_label.grid(row=6, column=0, padx=5, pady=5)
        self.generations_entry = ttk.Entry(self.root)
        self.generations_entry.grid(row=6, column=1, padx=5, pady=5)

        # Button to run the differential evolution algorithm
        self.run_button = ttk.Button(self.root, text="Run", command=self.run_differential_evolution)
        self.run_button.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

        # Canvas for plotting
        self.plot_canvas = tk.Canvas(self.root, width=600, height=400)
        self.plot_canvas.grid(row=8, column=0, columnspan=2, padx=5, pady=5)

    def run_differential_evolution(self):
        # Retrieve parameters from the GUI inputs
        x_strategy = self.strategy_var.get()
        y_vectors = int(self.vectors_entry.get())
        z_scheme = self.crossover_var.get()
        F_weight = float(self.F_entry.get())
        CR_probability = float(self.CR_entry.get())
        population_size = int(self.population_entry.get())
        max_generations = int(self.generations_entry.get())

        # Define the Ackley function
        def ackley_2d(x, y):
            return (20 + math.e
                    - 20 * math.exp(-.2 * (.5 * (x ** 2 + y ** 2)) ** .5)
                    - math.exp(.5 *
                               (math.cos(2 * math.pi * x)
                                + math.cos(2 * math.pi * y))))

        # Initialize DE with user-defined parameters
        de = DE(x=x_strategy, y=y_vectors, z=z_scheme, F=F_weight, CR=CR_probability)
        bound = 32.768
        pop = [[random.uniform(-bound, bound), random.uniform(-bound, bound)] for _ in range(population_size)]

        # Run DE
        best_solution, best_fitnesses = de.solve(ackley_2d, pop, iterations=max_generations)

        print("Best solution found:", best_solution)
        print("Best fitness:", ackley_2d(*best_solution))

        # Plot the best fitness over iterations
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, max_generations + 1), best_fitnesses)
        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness")
        plt.title("Best Fitness over Iterations")

        # Convert plot to Tkinter canvas
        self.plot_canvas.delete("all")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_img = tk.PhotoImage(data=buf.getvalue())
        self.plot_canvas.create_image(0, 0, anchor="nw", image=plot_img)

def main():
    root = tk.Tk()
    app = DifferentialEvolutionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
