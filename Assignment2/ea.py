import numpy as np
import matplotlib.pyplot as plt
from boosting_model import BoostingModel
import time

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


	def evaluate(self, model):
		# run model with parameter settings
		ndcg = model.run_on_genotype(self.genotype)

		# ndcg = int(''.join([str(_) for _ in self.genotype]), 2) / (2**(len(self.genotype)) - 1)
		# print(self.genotype)
		self.fitness = ndcg


	def set_genotype(self, genotype):
		self.genotype = genotype


	def __init__(self, size):
		self.n = size
		self.genotype = np.random.randint(2, size=self.n)
		self.fitness = None
		self.prob_mutate = 0.1
		self.eps = 0.01


class GeneticAlgorithm:

	def mutation(self, children):
		for i in range(len(children)):
			children[i].mutate()
			children[i].evaluate(self.boosting_model)

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
				# print(len(parents))
				parents.append(self.population[i])
				r = r + 1/self.lmbda
				# print('r: {}'.format(r))
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

		with open('fitness_over_time.txt', 'a') as f:
			fitnesses = [i.fitness for i in self.population]
			line = (str(fitnesses) + ', ' + str(time.ctime()) + '\n').replace('[', '').replace(']', '')
			f.write(line)


		with open('best_individual.txt', 'a') as f:
			for individual in self.population:
				f.write(str(self.generation) + ',' + ','.join([str(bit) for bit in individual.genotype ])+ '\n')



	def init_population(self):
		population = []

		for i in range(self.mu):
			genotype = Individual(self.n)
			genotype.evaluate(self.boosting_model)
			population.append(genotype)

		return population


	def update_best_individual(self):
		if len(self.fitnesses) == 0:
			self.best_genotype = self.population[0].genotype

		elif self.population[0].fitness > self.fitnesses[-1]:
			self.best_genotype = self.population[0].genotype

	def converged(self):
		if len(self.fitnesses) < 10:
			return False
		elif self.fitnesses[-1] / self.fitnesses[-5] >= 1.00001:
			return False
		else:
			print('------------converged------------, rate: {}, max_fitness: {}'.format(self.fitnesses[-1] / self.fitnesses[-2], str(self.fitnesses[-1])))
			return True


	def __init__(self):
		self.mu = 100
		self.lmbda = 50
		self.num_gen = 100
		self.n = 152
		# todo
		self.boosting_model = BoostingModel()
		self.population = self.init_population()
		self.fitnesses = []
		self.best_genotype = None
		self.generation = 0
		
		while self.generation < self.num_gen and not self.converged():
			print('generation {}'.format(self.generation))
			self.evolutionary_cycle()
			self.fitnesses.append(self.population[0].fitness)
			self.generation += 1

		


		print(self.best_genotype)
		plt.plot(self.fitnesses)
		plt.show()


GeneticAlgorithm()