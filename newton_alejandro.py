import Tarea1 as calculo
import golden_search_alejandro as gs
import numpy as np
from cauchy_alejandro import Cauchy

class Newton(Cauchy):
    def optimizar(self, funcion, x0):
        k = 0
        xk = np.array(x0, dtype = float)
        terminar = False
        calc = calculo.calculo(funcion, 0.001)

        while not terminar:
            grad = np.array(calc.Gradiente(xk))
            if np.linalg.norm(grad) <= self.epsilon1 or k >= self.M:
                terminar = True
            
            else:
                hess = np.array(calc.Hessiana(xk))
                hess_inv = np.linalg.inv(hess)
                direccion = hess_inv.dot(grad)
                def alpha_func(alpha):
                    return funcion.evaluar(xk - alpha * direccion)

                alpha = gs.GoldenSearch(self.epsilon2).buscar(alpha_func, 0, 1)
                xk_nuevo = xk - alpha * direccion
            
                if np.linalg.norm(xk_nuevo - xk)/(np.linalg.norm(xk) + 0.00001) <= self.epsilon1:
                    terminar = True
                
                else:
                    xk = xk_nuevo
                    k += 1

        return xk
# IRONEDIT:1781914815:f9b9aee97efc5861c858da6564e830138f0702224f58b993eabf3d63fc5ae479
