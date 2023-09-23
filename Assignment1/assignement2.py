import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose as mlr
import numpy as np


### 8-Queens problem

# def maxQueens(input):
#     fitness_cnt = 0

#     for i in range(len(input) - 1):
#         for j in range(i + 1, len(input)):
#             if (input[j] != input[i]) and (input[j] != input[i] + (j - i)) and (input[j] != input[i] - (j - i)):
#                 fitness_cnt += 1

#     return fitness_cnt

# fitness = mlr.CustomFitness(maxQueens)
# problem = mlr.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)

### Max-K colors

# EDGES = [(0,1),(0,3),(1,2),(1,3),(1,5),(2,3),(2,4),(3,4),(4,5)]
# def maxColors(input):
#     fitness_cnt = 0

#     for i in range(len(EDGES)):
#         if input[EDGES[i][0]] != input[EDGES[i][1]]:
#             fitness_cnt += 1

#     return fitness_cnt - len(set(input))

# fitness = mlr.CustomFitness(maxColors)
# problem = mlr.DiscreteOpt(length=6, fitness_fn=fitness, maximize=True, max_val=6)

### Travelling Salesman

CITIES = [(0, 0), (6, 2), (4, 2), (6, 7), (3, 2), (4, 8), (0, 5), (7, 3)]
def negativeDistance(input):
    fitness_cnt = -np.sqrt((CITIES[input[0]][0]-CITIES[input[-1]][0])**2+(CITIES[input[0]][1]-CITIES[input[-1]][1])**2)

    for i in range(len(input)-1):
        fitness_cnt -= np.sqrt((CITIES[input[i]][0]-CITIES[input[i+1]][0])**2+(CITIES[input[i]][1]-CITIES[input[i+1]][1])**2)

    return fitness_cnt

fitness = mlr.CustomFitness(negativeDistance, 'tsp')
problem = mlr.TSPOpt(length = 8, fitness_fn = fitness, maximize=True)

schedule = mlr.ExpDecay()
init_state = np.array(range(8))
# best_state, best_fitness = mlr.simulated_annealing(problem, schedule = schedule, max_attempts = 100, max_iters = 10000, init_state = init_state, random_state = 1)
# best_state, best_fitness = mlr.mimic(problem)
# best_state, best_fitness = mlr.random_hill_climb(problem)
best_state, best_fitness = mlr.genetic_alg(problem)
print(best_state)
print(best_fitness)
