"""
Command-line interface for genetic algorithm implementation for bohachevsky
function.
"""

import matplotlib.pyplot as plt

from genetic import Genetic


def main():
    """ Main function
    """
    while True:
        chromosome_size = int(input("Enter size of binary chromosome:"))
        if chromosome_size % 2 != 0:
            print("Chromosome size should be even, try again.")
            continue
        if chromosome_size < 2:
            print("Chromosome size should be greater than 1, try again.")
            continue
        break

    population_size = int(
        input("Enter size of population in each generation:")
    )

    generation_count = int(input("Enter number of generations:"))

    genetic = Genetic(chromosome_size, population_size, generation_count)

    while True:
        print("Enter crossover method:")
        print("1. n_point")
        print("2. single_point")
        print("3. uniform")
        crossover_method = ""
        crossover_method_number = int(input())
        if crossover_method_number == 1:
            while True:
                number_of_points = int(input("Enter n:"))
                if number_of_points < chromosome_size:
                    crossover_method = str(number_of_points) + "_point"
                    break
                print("Number of points should be less than chromosome size, try again.")
            break
        elif crossover_method_number == 2:
            crossover_method = "single_point"
            break
        elif crossover_method_number == 3:
            crossover_method = "uniform"
            break
        print("Wrong input, try again.")

    parent_selection_method = selection_input("parent selection")

    survival_selection_method = selection_input("survival selection")

    while True:
        mutation_selection_probability = float(input("Enter probability of selecting a chromosome for mutation:"))
        if mutation_selection_probability > 1 or mutation_selection_probability < 0:
            print("Probability of selecting a chromosome for mutation should be between 0 and 1, try again.")
            continue
        break

    while True:
        mutation_gene_probability = float(input("Enter probability of changing a gene for mutation:"))
        if mutation_gene_probability > 1 or mutation_gene_probability < 0:
            print("probability of changing a gene for mutation should be between 0 and 1, try again.")
            continue
        break

    genetic.run(
        crossover_method,
        parent_selection_method,
        survival_selection_method,
        mutation_selection_probability,
        mutation_gene_probability
    )

    print("Best answer ever:")
    print(f"Chromosome: {genetic.best_chromosome}")
    print(f"Fitness: {genetic.best_chromosome.fitness}")
    print(f"sumsq function output: {genetic.best_chromosome.calculate_sumsq_function()}")
    print(f"x: {genetic.best_chromosome.get_x()}")
    print(f"y: {genetic.best_chromosome.get_y()}")

    print("")
    print("Best answer in last iteration:")
    print(f"Chromosome: {genetic.best_chromosome_last_generation}")
    print(f"Fitness: {genetic.best_chromosome_last_generation.fitness}")
    print(f"sumsq function output: {genetic.best_chromosome_last_generation.calculate_sumsq_function()}")
    print(f"x: {genetic.best_chromosome_last_generation.get_x()}")
    print(f"y: {genetic.best_chromosome_last_generation.get_y()}")

    plt.plot(range(1, generation_count+1), genetic.generation_max_fitness)
    plt.title(label="Maximum fitness in different iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Maximum fitness")
    plt.show()

    plt.plot(range(1, generation_count+1), genetic.generation_average_fitness)
    plt.title(label="Average fitness in different iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average fitness")
    plt.show()


def selection_input(selection_type):
    """ Get selection method from input

    :param selection_type: Type of selection (parent selection or
    survival selection)
    :type selection_type: str
    """
    print(f"Enter {selection_type} method:")
    print("1. Roulette Wheel Selection (RWS)")
    print("2. Stochastic Universal Sampling (SUS)")
    print("3. Tournament Selection (TS)")
    if selection_type == "survival selection":
        print("4. Elitism")
    while True:
        selection_number = int(input())
        if selection_number == 1:
            return "rws"
        if selection_number == 2:
            return "sus"
        if selection_number == 3:
            return "ts_" + input("Enter size of tournament:")
        if selection_type == "survival selection":
            if selection_number == 4:
                return "elitism"
        print("Wrong input, try again.")


if __name__ == "__main__":
    main()
