import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import math
import matplotlib.pyplot as plt
from main import DE  # Assuming your DE algorithm is in a file named DE_algorithm.py


class DEGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Differential Evolution GUI")

        # Variables for storing user inputs
        self.x_strategy = tk.StringVar(value="rand")
        self.y_vectors = tk.IntVar(value=1)
        self.z_scheme = tk.StringVar(value="bin")
        self.F_weight = tk.DoubleVar(value=0.5)
        self.CR_probability = tk.DoubleVar(value=0.1)
        self.population_size = tk.IntVar(value=10)
        self.max_generations = tk.IntVar(value=100)

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Mutation strategy selection
        ttk.Label(self.root, text="Mutation Strategy:").grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(self.root, text="Random", variable=self.x_strategy, value="rand").grid(row=0, column=1,
                                                                                               sticky='w')
        ttk.Radiobutton(self.root, text="Best", variable=self.x_strategy, value="best").grid(row=0, column=2,
                                                                                             sticky='w')

        # Number of difference vectors
        ttk.Label(self.root, text="Number of Difference Vectors:").grid(row=1, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.y_vectors).grid(row=1, column=1, sticky='w')

        # Crossover scheme selection
        ttk.Label(self.root, text="Crossover Scheme:").grid(row=2, column=0, sticky='w')
        ttk.Radiobutton(self.root, text="Binomial", variable=self.z_scheme, value="bin").grid(row=2, column=1,
                                                                                              sticky='w')

        # Mutation weight (F)
        ttk.Label(self.root, text="Mutation Weight (F):").grid(row=3, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.F_weight).grid(row=3, column=1, sticky='w')

        # Crossover probability (CR)
        ttk.Label(self.root, text="Crossover Probability (CR):").grid(row=4, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.CR_probability).grid(row=4, column=1, sticky='w')

        # Population size
        ttk.Label(self.root, text="Population Size:").grid(row=5, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.population_size).grid(row=5, column=1, sticky='w')

        # Maximum generations
        ttk.Label(self.root, text="Maximum Generations:").grid(row=6, column=0, sticky='w')
        ttk.Entry(self.root, textvariable=self.max_generations).grid(row=6, column=1, sticky='w')

        # Solve button
        ttk.Button(self.root, text="Solve", command=self.solve_optimization).grid(row=7, column=0, columnspan=2,
                                                                                  pady=10)

    def solve_optimization(self):
        # Get user inputs
        x_strategy = self.x_strategy.get()
        y_vectors = self.y_vectors.get()
        z_scheme = self.z_scheme.get()
        F_weight = self.F_weight.get()
        CR_probability = self.CR_probability.get()
        population_size = self.population_size.get()
        max_generations = self.max_generations.get()

        # Run DE algorithm
        de = DE(x=x_strategy, y=y_vectors, z=z_scheme, F=F_weight, CR=CR_probability)
        bound = 10  # Adjust bound based on your requirement
        pop = [[random.uniform(-bound, bound) for _ in range(2)] for _ in range(population_size)]

        best_solution, best_fitnesses = de.solve(sum_of_squares, pop, iterations=max_generations)

        # Display results
        messagebox.showinfo("Optimization Results",
                            f"Best solution found: {best_solution}\nBest fitness: {sum_of_squares(*best_solution)}")

        # Plot the best fitness over iterations
        plt.plot(range(1, max_generations + 1), best_fitnesses)
        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness")
        plt.title("Best Fitness over Iterations")
        plt.show()


def sum_of_squares(*args):
    return sum(x ** 2 for x in args)


if __name__ == "__main__":
    root = tk.Tk()
    app = DEGUI(root)
    root.mainloop()
