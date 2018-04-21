import numpy as np
import matplotlib.pyplot as plt

class Individual:

	def mutate(self):
		m = np.random.uniform(size=self.n)
		m = m < self.prob_mutate
		
		for i in range(len(self.genotype)):
			if m[i]:
				if self.genotype[i] == 0:
					self.genotype[i] = 1
				else:
					self.genotype[i] = 0

	def evaluate(self):
		# run model with parameter settings

		self.fitness = np.sum(self.genotype)


	def set_genotype(self, genotype):
		self.genotype = genotype


	def __init__(self, size):
		self.n = size
		self.genotype = np.random.randint(2, size=self.n)
		self.fitness = None
		self.prob_mutate = 0.01


class GeneticAlgorithm:

	def mutation(self, children):
		for i in range(len(children)):
			children[i].mutate()
			children[i].evaluate()

		return children


	def uniform_crossover(self, p1, p2):
		# determine crossover point
		choice = np.random.uniform(size=self.n) < .5

		child1 = Individual(self.n)
		child2 = Individual(self.n)

		for i in range(self.n):
			if choice[i]:
				child1.genotype[i] = p1.genotype[i]
				child2.genotype[i] = p2.genotype[i]
			else:
				child1.genotype[i] = p2.genotype[i]
				child2.genotype[i] = p1.genotype[i]

		return child1, child2


	def recombination(self, parents):
		children = []
		if len(parents) % 2 != 0:
			raise Exception('Nr of parents should be even')

		for i in range(int(len(parents)/2)):
			p1 = parents[i*2]
			p2 = parents[i*2+1]

			c1, c2 = self.uniform_crossover(p1, p2) 

			children.append(c1)
			children.append(c2)

		return children


	def parent_selection(self):
		# fitness proportate selection using Stochastic Universal Sampling (SUS)
		fitnesses = [g.fitness for g in self.population]
		i = 0
		current = 0
		r = np.random.uniform(0, 1/self.lmbda)

		parents = []

		while current < self.lmbda:
			while r <= fitnesses[i]:
				parents.append(self.population[i])
				r = r + 1/self.lmbda
				current += 1

			i += 1

		return parents


		



	def survivor_selection(self, children):
		# use ranking based selection

		self.population.extend(children)

		self.population.sort(key=lambda x: x.fitness, reverse=True)

		# update fitness
		for individual in self.population:
			individual.evaluate()


		# return best 50 from population
		self.population = self.population[:self.mu]
		

	def evolutionary_cycle(self):
		# parent selection
		for individual in self.population:
			individual.evaluate()

		parents = self.parent_selection()

		# recombination
		children = self.recombination(parents)



		# mutation
		children = self.mutation(children)

		# survivor selection
		self.survivor_selection(children)

		self.update_best_individual()






	def init_population(self):
		population = []

		for i in range(self.mu):
			genotype = Individual(self.n)
			population.append(genotype)

		return population


	def update_best_individual(self):
		if len(self.fitnesses) == 0:
			self.best_genotype = self.population[0].genotype

		elif self.population[0].fitness > self.fitnesses[-1]:
			self.best_genotype = self.population[0].genotype



	def __init__(self):
		self.mu = 100
		self.lmbda = 20
		self.num_gen = 20
		self.n = 18
		self.population = self.init_population()
		self.fitnesses = []
		self.best_genotype = None

		for i in range(self.num_gen):
			print('generation {}'.format(i))
			self.evolutionary_cycle()

			self.fitnesses.append(self.population[0].fitness)

		print(self.best_genotype)
		plt.plot(self.fitnesses)
		plt.show()



GeneticAlgorithm()
