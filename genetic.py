# -*- coding: utf-8 -*-
"""
Spyder Editor

Algoritmo Genético - Henrique Voni
"""

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

## Classe de cidade
class City:
    
    ## Inicializa classe
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    ## Calcula distância euclidiana
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    ## Coordenadas da cidade
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
## Classe de fitness
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    def routeDistance(self):
            
        if self.distance == 0:
            pathDistance = 0    
            ## Percorre as cidades
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                
                ## Se não tem cidade na rota, adiciona
                if i+1 < len(self.route):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0]
                
                ## acrescenta distancia do caminho
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
            print("Fitness: ", self.fitness)
        return self.fitness


## Criar população de rotas    
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


## Criar uma população
def initialPopulation(popSize, cityList):
    population = []
    
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population



## Ranquear o fitness de cada rota
def rankRoutes(population):
    fitnessResults = {}
    
    ## loop para calcular cada fitness
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    
    ## retorna ordenado os fitness
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse=True)



## Seleção dos indivíduos
def selection(popRanked, eliteSize):
    selectionResults = []
    
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    
    ## Método da roleta
    ## Soma acumulativa dos fitness
    df['cum_sum'] = df.Fitness.cumsum()
    
    ## Define porcentagem de cada rota dividindo pelo total
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    
    ## Seleciona elite
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    
    ## Seleciona demais indivíduos pelo metodo da roleta
    for i in range(0, len(popRanked) - eliteSize):
        
        ## Chance de seleção
        pick = 100 * random.random()

        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
            
    return selectionResults

## Mating Pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


## Cruzamento
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    ## Inicio e fim do corte de cruzamento
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    
    childP2 = [item for item in parent2 if item not in childP1]
    
    child = childP1 + childP2
    return child


## Nova populacao
def breedPopulation(matingPool, eliteSize):
    children = []
    length = len(matingPool) - eliteSize
    pool = random.sample(matingPool, len(matingPool))
    
    for i in range(0, eliteSize):
        children.append(matingPool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingPool) - i - 1])
        children.append(child)
    return children

## mutação
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        # SE aleatorio menor que taxa de mutacão, troca
        if(random.random() < mutationRate):
            
            #seleciona indice 
            swapWith = int(random.random() * len(individual))
            
            #permuta 2 cidades no individuo
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
            
    return individual

## Gera mutação na população
def mutatePopulation(population, mutationRate):
    
    mutatedPop = []
    
    ## aplica a mutação em cada individuo da população
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    
    return mutatedPop


## Define nova geração
def nextGeneration(currentGen, eliteSize, mutationRate):
    
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    
    return nextGeneration

## algoritmo genetico
def ga(population, popSize, eliteSize, mutationRate, generations):
    
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute



## Exemplo
cityList = []

## criar 25 cidades aleatórias
for i in range(0, 25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))


oneRoute = ga(population=cityList, popSize=100, eliteSize=20, mutationRate = 0.01, generations=500)

    
def showMap(cityList):
    print(cityList)
    prev=City(0,0)
    for i in cityList:
        plt.plot(i.x, i.y,'ro')
        plt.plot(prev.x,prev.y, 'k-')
        if(prev.x == 0 and prev.y == 0):
            prev=i
            continue;
        else:
            plt.plot([prev.x,i.x],[prev.y, i.y],'k-')
            prev=i
    plt.show()

showMap(oneRoute)



