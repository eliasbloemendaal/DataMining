import numpy as np

class individual:

	def mutate(self):
		m = np.random.uniform(size=self.n)
		m = m < self.prob_mutate
		
		for i in len(child):
			if m[i]:
				if self.genotype[i] == 0:
					self.genotype[i] = 1
				else:
					self.genotype[i] = 0

	def evaluate(self):
		# run model with parameter settings

		self.fitness = np.sum(self.genotype)


	def __init__(self, size):
		self.n = size
		self.genotype = np.random.randint(2, size=self.n)
		self.fitness = None




class genetic_algorithm:




	def mutation(self, children):
		for i in len(children):
			children[i].mutate()

		return children


	def uniform_crossover(self, p1, p2):
		# determine crossover point
		choice = np.random.uniform(size=self.n) < .5

		child1 = np.zeros(self.n)
		child2 = np.zeros(self.n)

		for i in range(self.n):
			if choice[i]:
				child1[i] = p1[i]
				child2[i] = p2[i]
			else:
				child1[i] = p2[i]
				child2[i] = p1[i]

		return child1, child2


	def recombination(self, parents):
		children = []
		if len(parents) % 2 != 0:
			raise Exception('Nr of parents should be even')

		for i in range(len(parents)/2):
			p1 = parents[i*2]
			p2 = parents[i*2+1]

			c1, c2 = uniform_crossover(p1, p2) 

			children.append(c1)
			children.append(c2)



		return children


	def parent_selection(self):
		# fitness proportate selection using Stochastic Universal Sampling (SUS)



	def survivor_selection(self, children):
		# use ranking based selection
		self.population.sort(key=lambda x: x.fitness, reverse=True)

		# update fitness
		for individual in self.population:
			individual.evaluate()

		# return best 50 from population
		parents = self.population[:50]

		return parents
		

	def evolutionary_cycle(self):
		# parent selection
		parents = self.parent_selection()

		# recombination
		children = self.recombination(parents)

		# mutation
		children = self.mutation(children)

		# survivor selection
		new_gen = self.survivor_selection(children)






	def init_population(self):
		population = []

		for i in range(self.mu):
			genotype = individual(self.n)
			population.append(genotype)

		return population




	def __init__(self):
		self.mu = 100
		self.num_gen = 10
		self.n = 18
		self.prob_mutate = 0.01
		self.population = self.init_population()

		for i in range(self.num_gen):
			self.evolutionary_cycle()
