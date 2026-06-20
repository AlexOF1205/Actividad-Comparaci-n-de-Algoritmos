import Tarea1 as calculo
import golden_search_alejandro as gs
import numpy as np

class GradienteConjugado:
    def  __init__(self, epsilon1, epsilon2, epsilon3, M):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.M = M

    def optimizar(self, funcion, x0):
        calc = calculo.calculo(funcion, 0.001)

        xk = np.array(x0, dtype=float)
        grad = np.array(calc.Gradiente(xk))
        s = -grad
        k = 0
        terminar = False

        while not terminar and k < self.M:
            def alpha_func(alpha):
                return funcion.evaluar(xk + alpha * s)
                
            alpha = gs.GoldenSearch(self.epsilon1).buscar(alpha_func, 0, 1)
            xk_nuevo = xk + alpha * s

            grad_nuevo = np.array(calc.Gradiente(xk_nuevo))
        
            cambio_relativo = np.linalg.norm(xk_nuevo - xk)/(np.linalg.norm(xk) + 0.00001)
            if cambio_relativo <=self.epsilon2 or np.linalg.norm(grad_nuevo) <= self.epsilon3:
                terminar = True
                xk = xk_nuevo
            else:
                beta = (np.linalg.norm(grad_nuevo)**2)/(np.linalg.norm(grad)**2)
                s = -grad_nuevo + beta * s
                xk = xk_nuevo
                grad = grad_nuevo
                k += 1

        return xk
# IRONEDIT:1781916329:8888f7cefd6103ff1c3e887cdec4fee7e317f6036709e2e8f82f22c82ce3d886
