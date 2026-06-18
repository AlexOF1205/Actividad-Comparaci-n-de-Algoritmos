import Tarea1 as calculo
import golden_search_alejandro as gs
import numpy as np

class Cauchy:
    def __init__(self, epsilon1, epsilon2, M):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.M = M

    def optimizar(self, funcion, x0):
        k = 0
        xk = np.array(x0)
        terminar = False
        calc = calculo.calculo(funcion, 0.001)

        while not terminar:
            grad = np.array(calc.Gradiente(xk))
            if np.linalg.norm(grad) <= self.epsilon1 or k >= self.M:
                terminar = True
            
            else:
                def alpha_func(alpha):
                    return funcion.evaluar(xk - alpha * grad)

                alpha = gs.GoldenSearch(self.epsilon2).buscar(alpha_func, 0, 1)
                xk_nuevo = xk - alpha * grad
            
                if np.linalg.norm(xk_nuevo - xk)/(np.linalg.norm(xk) + 0.00001) <= self.epsilon1:
                    terminar = True
                
                else:
                    xk = xk_nuevo
                    k += 1

        return xk
# IRONEDIT:1781748201:e099c5187304064b74a53b4cba84809b896f1e4127eaee014dce0572f66ba786
