"""
Genetic algorithm implementation for Ackley function.
"""

import random

from chromosome import Chromosome


class Genetic:
    """ Genetic algorithm for Ackley function

    """

    def __init__(self, chromosome_size, population_size, generation_count):
        """ Initialize an instance of genetic class.
        :param chromosome_size: Size of each binary chromosome.
        :type chromosome_size: int
        :param population_size: Size of each generation population
        :type population_size: int
        :param generation_count: Number of generations
        :type generation_count: int
        """
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.generation_count = generation_count
        self.current_generation = self.initialize_population()
        self.next_generation = list()
        self.generation_max_fitness = list()
        self.generation_average_fitness = list()
        self.best_chromosome = Genetic.find_best_chromosome(
            self.current_generation
        )
        self.best_chromosome_last_generation = Genetic.find_best_chromosome(
            self.current_generation
        )

    def initialize_population(self):
        """ Initialize random population.

        :return: List of random chromosome
        :rtype: list
        """
        return [
            Chromosome.random(self.chromosome_size)
            for _ in range(self.population_size)
        ]

    def run(
            self, crossover_method="3_point",
            parent_selection_method="rws",
            survival_selection_method="rws",
            mutation_selection_probability=1.0,
            mutation_gene_probability=0.1
    ):
        """ Run genetic algorithm for ackley function in given methods.

        :param crossover_method: Crossover method for binary chromosome, could
        be "n_point" (a number instead of n), "single_point" or "uniform",
        defaults to "3_point"
        :type crossover_method: str, optional
        :param parent_selection_method: Parent selection method, could be "rws"
        (Roulette Wheal Selection), "sus" (Stochastic Universal Sampling),
        "ts_n" (Tournament Selection with size n) or "rb" (Rank-based Selection
        ), defaults to "rws"
        :type parent_selection_method: str, optional
        :param survival_selection_method: Survival selection method, could be
        "rws" (Roulette Wheal Selection), "sus" (Stochastic Universal Sampling)
        , "ts_n" (Tournament Selection with size n), "rb" (Rank-based
        Selection) or "elitism", defaults to "rws"
        :type survival_selection_method: str, optional
        :param mutation_selection_probability: Probability of selecting a
        chromosome to perform mutation, defaults to 1.0
        :type mutation_selection_probability: float, optional
        :param mutation_gene_probability: Probability of changing a gene in
        mutation, defaults to 0.1
        :type mutation_gene_probability: float, optional
        """
        for _ in range(self.generation_count):
            self.generation_max_fitness.append(
                max(self.current_generation, key=lambda c: c.fitness).fitness
            )
            self.generation_average_fitness.append(
                Genetic.fitness_sum(self.current_generation)
                / self.population_size
            )
            selected_parent_list = self.parent_selection(
                parent_selection_method
            )
            for i in range(0, len(selected_parent_list), 2):
                offspring_list = []
                offspring_list.extend(
                    selected_parent_list[i].crossover(
                        selected_parent_list[i+1],
                        crossover_method
                    )
                )
                offspring_list[0].mutation(
                    mutation_selection_probability,
                    mutation_gene_probability
                )
                offspring_list[1].mutation(
                    mutation_selection_probability,
                    mutation_gene_probability
                )
                self.next_generation.extend(
                    self.survival_selection(
                        offspring_list + [
                            selected_parent_list[i],
                            selected_parent_list[i+1]
                        ],
                        survival_selection_method
                    )
                )
            self.go_to_the_future()
            self.best_chromosome = max(
                self.best_chromosome,
                Genetic.find_best_chromosome(self.current_generation),
                key=lambda c: c.fitness
            )
        self.best_chromosome_last_generation = Genetic.find_best_chromosome(
            self.current_generation
        )

    def parent_selection(self, parent_selection_method):
        """ Parent selection based on selection method.

        :param parent_selection_method: Parent selection method (rws, sus, ts,
        rb)
        :type parent_selection_method: str
        :return: List of chromosomes, to be the parents of the next generation
        :rtype: list
        """
        return Genetic.selection(
            self.current_generation,
            self.population_size,
            parent_selection_method
        )

    def go_to_the_future(self):
        """ Change current generation to the next generation.

        :return: NoneType
        :rtype: NoneType
        """
        self.current_generation = self.next_generation
        self.next_generation = list()

    @staticmethod
    def survival_selection(chromosome_list, survival_selection_method):
        """ Survival selection based on selection method.

        :param chromosome_list: Chromosome list to select from
        :type chromosome_list: list
        :param survival_selection_method: Survival selection method (rws, sus,
        ts, rb)
        :type survival_selection_method: str
        :return: List of chromosomes, to be the survivals of the next
        generation
        :rtype: list
        """
        return Genetic.selection(
            chromosome_list,
            2,
            survival_selection_method
        )

    @staticmethod
    def selection(chromosome_list, size, selection_method):
        """ Select chromosomes based on selection method.

        :param chromosome_list: Chromosome list to select from
        :type chromosome_list: list
        :param size: Size of chromosomes that should be selected
        :type size: int
        :param selection_method: Selection method (rws, sus, ts, rb, elitism)
        :type selection_method: str
        :return: List of selected chromosomes
        :rtype: list
        """
        if selection_method == "rws":
            return [
                Genetic.roulette_wheal_selection(chromosome_list)
                for _ in range(size)
            ]
        if selection_method == "rb":
            return [
                Genetic.rank_based_selection(chromosome_list)
                for _ in range(size)
            ]
        if selection_method[0:2] == "ts":
            return [
                Genetic.tournament_selection(
                    chromosome_list,
                    int(selection_method.split('_')[1])
                )
                for _ in range(size)
            ]
        if selection_method == "sus":
            return Genetic.stochastic_universal_sampling(chromosome_list, size)
        if selection_method == "elitism":
            chromosome_list.sort(reverse=True, key=lambda c: c.fitness)
            return [
                chromosome_list[i] for i in range(size)
            ]
        return False

    @staticmethod
    def fitness_sum(chromosome_list):
        """ Add up all the chromosomes fitness in current generation.

        :return: Sum of all chromosomes fitness in current generation
        :rtype: float
        """
        summation = 0
        for chromosome in chromosome_list:
            summation += chromosome.fitness
        return summation

    @staticmethod
    def rank_based_selection(chromosome_list):
        """ Select a chromosome based on rank selection (rb)

        :param chromosome_list: List of chromosome
        :type chromosome_list: list
        :return: Selected chromosome based on rank selection
        :rtype: Chromosome
        """
        chromosome_list.sort(key=lambda c: c.fitness)
        rank_sum = ((1 + len(chromosome_list)) * len(chromosome_list)) / 2
        random_float = random.uniform(0, rank_sum)
        for i, chromosome in enumerate(chromosome_list):
            random_float -= i + 1
            if random_float <= 0:
                return chromosome
        return chromosome_list[-1]

    @staticmethod
    def roulette_wheal_selection(chromosome_list):
        """ Select a chromosome based on roulette wheal selection (rws)

        :param chromosome_list: List of chromosome
        :type chromosome_list: list
        :return: Selected chromosome based on roulette wheal selection
        :rtype: Chromosome
        """
        fitness_sum = Genetic.fitness_sum(chromosome_list)
        random_float = random.uniform(0, fitness_sum)
        for chromosome in chromosome_list:
            random_float -= chromosome.fitness
            if random_float <= 0:
                return chromosome
        return chromosome_list[-1]

    @staticmethod
    def tournament_selection(chromosome_list, size):
        """ Select a chromosome based on tournament selection (ts_n)

        :param chromosome_list: List of chromosome
        :type chromosome_list: list
        :param size: Size of tournament
        :type size: int
        :return: Selected chromosome based on tournament selection
        :rtype: Chromosome
        """
        tournament = random.sample(chromosome_list, k=size)
        return max(tournament, key=lambda c: c.fitness)

    @staticmethod
    def stochastic_universal_sampling(chromosome_list, size):
        """ Select a list of chromosome based on stochastic universal sampling
        (sus)

        :param chromosome_list: List of chromosome
        :type chromosome_list: list
        :param size: Size of chromosome list to be selected
        :type size: int
        :return: Selected chromosome based on stochastic universal sampling
        :rtype: Chromosome
        """
        fitness_sum = Genetic.fitness_sum(chromosome_list)
        random_float = random.uniform(0, fitness_sum)
        distance = fitness_sum / size
        selected_chromosome_list = list()
        while True:
            for chromosome in chromosome_list:
                random_float -= chromosome.fitness
                if random_float <= 0:
                    selected_chromosome_list.append(chromosome)
                    if len(selected_chromosome_list) == size:
                        return selected_chromosome_list
                    random_float = distance + random_float
        return random.shuffle(selected_chromosome_list)

    @staticmethod
    def find_best_chromosome(chromosome_list):
        """ Find best chromosome from a chromosome list.

        :param chromosome_list: List of chromosome
        :type chromosome_list: list
        :return: Best chromosome from chromosome_list
        :rtype: Chromosome
        """
        return max(chromosome_list, key=lambda c: c.fitness)
