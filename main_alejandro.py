import numpy as np
import time

import cauchy_alejandro as c
import newton_alejandro as n
import canjugado_alejandro as gc

class Rastrigin:
    def evaluar(self, x):
        A = 10
        total = A* len(x)
        for xi in x:
            total += xi**2 - A * np.cos(2*np.pi*xi)
        return total

class Rosenbrock:
    def evaluar(self, x):
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return total

x0 = [-2.0, -2.0, -2.0]
epsilon1 = 0.001
epsilon2 = 0.001
epsilon3 = 0.001

M = 100

ras = Rastrigin()
ros = Rosenbrock()

ca = c.Cauchy(epsilon1, epsilon2, M)
ne = n.Newton(epsilon1, epsilon2, M)
co = gc.GradienteConjugado(epsilon1, epsilon2, epsilon3, M)

print("RASTRIGIN")
print("Cauchy: ",ca.optimizar(ras, x0)) 
print("Newton: ",ne.optimizar(ras, x0)) 
print("Conjugado: ",co.optimizar(ras, x0)) 

print("ROSENBROCK")
print("Cauchy: ",ca.optimizar(ros, x0)) 
print("Newton: ",ne.optimizar(ros, x0)) 
print("Conjugado: ",co.optimizar(ros, x0))
# IRONEDIT:1781917322:c830aa7d41b8a26568ca20225fce8df4703724c56dab2595e4233fcd0fd75e87
