"""
Binary chromosome representation with dynamic size and all 3 model of crossover
single-point, n-point and uniform. Each chromosome could be initialized by
calling random or form_gen_list classmethod.
"""

import random
import math


class Chromosome():
    """ Representation of a chromosome in both phenotype and genotype.

    """
    def __init__(self, genotype, minimum=-5, maximum=5):
        self.size = len(genotype)
        self.minimum = minimum
        self.maximum = maximum
        self.genotype = genotype
        self.fitness = self.calculate_fitness()

    @classmethod
    def random(cls, size, minimum=-5, maximum=5):
        """ Create a random chromosome instance.

        :param size: Size of chromosome
        :type size: int
        :param minimum: Minimum value in phenotype, defaults to -5
        :type minimum: int, optional
        :param maximum: Maximum value in phenotype, defaults to 5
        :type maximum: int, optional
        :return: An instance of chromosome class
        :rtype: Chromosome
        """
        if size % 2 == 1:
            print("size can't be odd, increasing by 1 automatically")
            size += 1
        genotype = []
        for _ in range(size):
            genotype.append(bool(random.getrandbits(1)))
        return cls(genotype, minimum, maximum)

    @classmethod
    def from_gen_list(cls, gen_list, minimum=-5, maximum=5):
        """ Create a chromosome based on a gen_list.

        :param gen_list: A list which the chromosome should be created from
        :type gen_list: list
        :param minimum: Minimum value in phenotype, defaults to -5
        :type minimum: int, optional
        :param maximum: Maximum value in phenotype, defaults to 5
        :type maximum: int, optional
        :return: An instance of chromosome class
        :rtype: Chromosome
        """
        genotype = [bool(int(x)) for x in gen_list]
        return cls(genotype, minimum, maximum)

    def __str__(self):
        return ''.join(['1' if x else '0' for x in self.genotype])

    def __repr__(self):
        return ''.join(['1' if x else '0' for x in self.genotype])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [1 if x else 0 for x in self.genotype[key]]
        return 1 if self.genotype[key] else 0

    def __setitem__(self, key, value):
        self.genotype[key] = bool(int(value))

    def __calculate_phenotype_value(self, gen_list):
        raw_value = 0
        for i, gen in enumerate(gen_list):
            raw_value += gen * (2 ** i)
        return (
            raw_value *
            ((self.maximum - self.minimum) / ((2 ** len(gen_list)) - 1))
            + self.minimum
        )

    def __get_random_split_point_list(self, count):
        """ Generate random unique split points.

        :param count: Number of split points needed.
        :type count: int
        :return: Sorted list of random split points
        :rtype: list
        """
        split_point_list = random.sample(range(1, self.size), count)
        split_point_list.sort()
        split_point_list.insert(0, 0)
        split_point_list.append(self.size)
        return split_point_list

    def calculate_ackley_function(self):
        """ Calculate ackley function with x and y extracted from chromosome.

        :return: Ackley function output
        :rtype: float
        """
        ackley_x = self.get_x()
        ackley_y = self.get_y()
        power1 = -0.2 * math.sqrt(0.5 * ((ackley_x ** 2) + (ackley_y ** 2)))
        power2 = +0.5 * (
            math.cos(2 * math.pi * ackley_x) + math.cos(2 * math.pi * ackley_y)
        )
        return -20 * math.exp(power1) - math.exp(power2) + math.e + 20

    def calculate_fitness(self, base=21):
        """ Calculate fitness by subtracting Ackley function value from 21,
        since maximum value of ackley function is 20, our worst chromosome
        fitness is 1 and every other chromosome is higher than 1.

        :param base: Base of our fitness calculation, defaults to 21
        :type base: int, optional
        :return: Fitness of the chromosome
        :rtype: float
        """
        return base - self.calculate_ackley_function()

    def general_crossover(self, second_parent, split_point_list):
        """ Crossover with two parent chromosome and produce two children.

        :param second_parent: Second parent
        :type second_parent: Chromosome
        :param split_point_list: A list with random split points
        :type split_point_list: list
        :return: A tuple containing two children
        :rtype: tuple
        """
        first_parent = self
        first_child = []
        second_child = []
        for i, (start, end) in enumerate(
                zip(split_point_list[:-1], split_point_list[1:])
        ):
            if i % 2 == 0:
                first_child[start:end] = first_parent[start:end]
                second_child[start:end] = second_parent[start:end]
            else:
                first_child[start:end] = second_parent[start:end]
                second_child[start:end] = first_parent[start:end]
        return (
            self.from_gen_list(first_child), self.from_gen_list(second_child)
        )

    def single_point_crossover(self, second_parent):
        """ Single point crossover.

        :param second_parent: Second parent
        :type second_parent: Chromosome
        :return: A tuple containing two children
        :rtype: tuple
        """
        split_point_list = self.__get_random_split_point_list(1)
        return self.general_crossover(second_parent, split_point_list)

    def n_point_crossover(self, second_parent, count):
        """ N-Point crossover.

        :param second_parent: Second parent
        :type second_parent: Chromosome
        :param count: Number of split points
        :type count: int
        :return: A tuple containing two children
        :rtype: tuple
        """
        split_point_list = self.__get_random_split_point_list(count)
        return self.general_crossover(second_parent, split_point_list)

    def uniform_crossover(self, second_parent):
        """ Uniform crossover, each bit is chosen from either parent.

        :param second_parent: Second parent
        :type second_parent: Chromosome
        :return: A tuple containing two children
        :rtype: tuple
        """
        first_parent = self
        first_child = []
        second_child = []
        for i, j in zip(first_parent, second_parent):
            if bool(random.getrandbits(1)):
                first_child.append(i)
                second_child.append(j)
            else:
                first_child.append(j)
                second_child.append(i)
        return (
            self.from_gen_list(first_child), self.from_gen_list(second_child)
        )

    def crossover(self, second_parent, method):
        """ Crossover with chosen method (n_point, single_point, uniform).

        :param second_parent: Second parent
        :type second_parent: Chromosome
        :param method: Method for crossover
        :type method: str
        """
        if method == "uniform":
            return self.uniform_crossover(second_parent)
        if method.split('_')[0] == "single":
            return self.single_point_crossover(second_parent)
        if method.split('_')[0].isnumeric():
            return self.n_point_crossover(
                second_parent, int(method.split('_')[0])
            )
        return False

    def mutation(self, selection_probability, gene_probability):
        """ Perform mutation on chromosome

        :param selection_probability: Probability of selecting a chromosome
        :type selection_probability: float
        :param gene_probability: Probability of changing each gene
        :type gene_probability: float
        """
        if random.random() > selection_probability:
            return
        for i in range(self.size):
            if random.random() <= gene_probability:
                self.genotype[i] = not self.genotype[i]
        self.fitness = self.calculate_fitness()

    def get_x(self):
        """ Get value of x in phenotype space.

        :return: Value of x
        :rtype: float
        """
        return self.__calculate_phenotype_value(self[:int(self.size/2)])

    def get_y(self):
        """ Get value of y in phenotype space.

        :return: Value of y
        :rtype: float
        """
        return self.__calculate_phenotype_value(self[int(self.size/2):])
