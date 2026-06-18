import math

class GoldenSearch:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.phi = (math.sqrt(5)-1)/2

    def buscar(self, funcion, a, b):
        dif = b - a
        while dif > self.epsilon:
            x1 = b - self.phi*dif
            x2 = a + self.phi*dif

            fx1 = funcion(x1)
            fx2 = funcion(x2)
            
            if fx1 > fx2:
                a = x1
            elif fx1 < fx2:
                b = x2
            else:
                a = x1
                b = x2
            
            dif = b-a
        return (a + b)/2
# IRONEDIT:1781745909:9a8b2e91ab105ec93a492ca046aa7fe028f60d7fdf67542d381b828471d3020f
