import numpy as np
import matplotlib.pyplot as plt
from model import lstm

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

	def evaluate(self, lstm):
		# run model with parameter settings
		mse = lstm.run_on_genotype(self.genotype)

		mean_mse = np.mean(np.array(mse))

		# mean_mse = np.sum(self.genotype)



		self.fitness = 1 / (self.eps + mean_mse)
		# self.fitness = np.sum(self.genotype)


	def set_genotype(self, genotype):
		self.genotype = genotype


	def __init__(self, size):
		self.n = size
		self.genotype = np.random.randint(2, size=self.n)
		self.fitness = None
		self.prob_mutate = 0.01
		self.eps = 0.01


class GeneticAlgorithm:

	def mutation(self, children):
		for i in range(len(children)):
			children[i].mutate()
			children[i].evaluate(self.lstm)

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
		# print(parents)
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
		print(len(fitnesses))
		a = [sum(fitnesses[:i+1])/sum(fitnesses) for i in range(len(fitnesses))]
		i = 0
		current = 0
		r = np.random.uniform(0, 1/self.lmbda)

		parents = []

		while current < self.lmbda:
			while r < a[i]:
				print(len(parents))
				parents.append(self.population[i])
				r = r + 1/self.lmbda
				print('r: {}'.format(r))
				current += 1

			i += 1
			
		print(len(parents))
		return parents


		



	def survivor_selection(self, children):
		# use ranking based selection

		self.population.extend(children)

		self.population.sort(key=lambda x: x.fitness, reverse=True)

		# # update fitness
		# for individual in self.population:
		# 	individual.evaluate(self.lstm)


		# return best 50 from population
		self.population = self.population[:self.mu]
		

	def evolutionary_cycle(self):
		parents = self.parent_selection()

		# recombination
		children = self.recombination(parents)

		# mutation
		children = self.mutation(children)

		# survivor selection
		self.survivor_selection(children)

		self.update_best_individual()

		with open('results.txt', 'a') as f:
			fitnesses = [i.fitness for i in self.population]
			line = (str(fitnesses) + '\n').replace('[', '').replace(']', '')
			f.write(line)






	def init_population(self):
		population = []

		for i in range(self.mu):
			genotype = Individual(self.n)
			genotype.evaluate(self.lstm)
			population.append(genotype)

		return population


	def update_best_individual(self):
		if len(self.fitnesses) == 0:
			self.best_genotype = self.population[0].genotype

		elif self.population[0].fitness > self.fitnesses[-1]:
			self.best_genotype = self.population[0].genotype



	def __init__(self):
		self.mu = 4
		self.lmbda = 4
		self.num_gen = 3
		self.n = 44
		self.lstm = lstm()
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