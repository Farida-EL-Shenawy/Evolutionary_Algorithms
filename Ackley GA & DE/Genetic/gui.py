import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random

from genetic import Genetic

class GeneticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm for Ackley Function")

        self.create_widgets()

    def create_widgets(self):
        # Chromosome size
        self.chromosome_size_label = ttk.Label(self.root, text="Chromosome Size:")
        self.chromosome_size_label.grid(row=0, column=0, padx=5, pady=5)
        self.chromosome_size_entry = ttk.Entry(self.root)
        self.chromosome_size_entry.grid(row=0, column=1, padx=5, pady=5)

        # Population size
        self.population_size_label = ttk.Label(self.root, text="Population Size:")
        self.population_size_label.grid(row=1, column=0, padx=5, pady=5)
        self.population_size_entry = ttk.Entry(self.root)
        self.population_size_entry.grid(row=1, column=1, padx=5, pady=5)

        # Generation count
        self.generation_count_label = ttk.Label(self.root, text="Generation Count:")
        self.generation_count_label.grid(row=2, column=0, padx=5, pady=5)
        self.generation_count_entry = ttk.Entry(self.root)
        self.generation_count_entry.grid(row=2, column=1, padx=5, pady=5)

        # Crossover method
        self.crossover_label = ttk.Label(self.root, text="Crossover Method:")
        self.crossover_label.grid(row=3, column=0, padx=5, pady=5)
        self.crossover_var = tk.StringVar(self.root)
        self.crossover_var.set("single_point")  # Default value
        self.crossover_menu = ttk.OptionMenu(
            self.root, self.crossover_var, "single_point", "single_point", "n_point", "uniform"
        )
        self.crossover_menu.grid(row=3, column=1, padx=5, pady=5)

        # Parent selection method
        self.parent_selection_label = ttk.Label(self.root, text="Parent Selection:")
        self.parent_selection_label.grid(row=4, column=0, padx=5, pady=5)
        self.parent_selection_var = tk.StringVar(self.root)
        self.parent_selection_var.set("rws")  # Default value
        self.parent_selection_menu = ttk.OptionMenu(
            self.root, self.parent_selection_var, "rws", "rws", "sus", "ts_2", "rb"
        )
        self.parent_selection_menu.grid(row=4, column=1, padx=5, pady=5)

        # Survival selection method
        self.survival_selection_label = ttk.Label(self.root, text="Survival Selection:")
        self.survival_selection_label.grid(row=5, column=0, padx=5, pady=5)
        self.survival_selection_var = tk.StringVar(self.root)
        self.survival_selection_var.set("rws")  # Default value
        self.survival_selection_menu = ttk.OptionMenu(
            self.root, self.survival_selection_var, "rws", "rws", "sus", "ts_2", "rb", "elitism"
        )
        self.survival_selection_menu.grid(row=5, column=1, padx=5, pady=5)

        # Button to run the genetic algorithm
        self.run_button = ttk.Button(self.root, text="Run", command=self.run_genetic_algorithm)
        self.run_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        # Canvas for plotting
        self.plot_canvas = tk.Canvas(self.root, width=600, height=400)
        self.plot_canvas.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

    def run_genetic_algorithm(self):
        chromosome_size = int(self.chromosome_size_entry.get())
        population_size = int(self.population_size_entry.get())
        generation_count = int(self.generation_count_entry.get())
        crossover_method = self.crossover_var.get()
        parent_selection_method = self.parent_selection_var.get()
        survival_selection_method = self.survival_selection_var.get()

        genetic = Genetic(chromosome_size, population_size, generation_count)
        genetic.run(crossover_method=crossover_method,
                    parent_selection_method=parent_selection_method,
                    survival_selection_method=survival_selection_method)

        self.plot_fitness(genetic.generation_max_fitness)

    def plot_fitness(self, fitness_data):
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(fitness_data) + 1), fitness_data)
        plt.title("Maximum Fitness in Different Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Maximum Fitness")

        # Convert plot to Tkinter canvas
        self.plot_canvas.delete("all")
        plt.savefig("plot.png")
        plt.close()
        self.plot_img = tk.PhotoImage(file="plot.png")
        self.plot_canvas.create_image(0, 0, anchor="nw", image=self.plot_img)

    def plot_avg_fitness(self,fitness_data):
        plt.plot(range(1, generation_count + 1), genetic.generation_average_fitness)
        plt.title(label="Average fitness in different iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Average fitness")

        self.plot_canvas.delete("all")
        plt.savefig("plot.png")
        plt.close()
        self.plot_img = tk.PhotoImage(file="plot.png")
        self.plot_canvas.create_image(0, 0, anchor="nw", image=self.plot_img)


def main():
    root = tk.Tk()
    app = GeneticGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
