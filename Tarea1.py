 # Librerías
import math
import numpy as np

# Funciones
class F1:
    def evaluar(self, x):
        return x[0]**2-2*x[1]
class F2:
    def evaluar(self, x):
        return x[0]**2+x[1]*x[2]-x[2]**2
class F3:
    def evaluar(self, x):
        return x[0]**2+2*x[1]**2+3*x[2]**2+x[0]*x[3]

# Operaciones
class calculo:
    def __init__(self, funcion, delta):
        self.funcion = funcion
        self.delta = delta

    def Gradiente(self, x):
        guardado = []
        for i in range(len(x)):
            x_mas = x.copy()
            x_menos = x.copy()
            
            x_mas[i] += self.delta
            x_menos[i] -= self.delta

            Grad = (self.funcion.evaluar(x_mas)-self.funcion.evaluar(x_menos))/(2*self.delta)

            guardado.append(Grad)
        return guardado

    def Hessiana(self, x):
        guardado = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j:
                    x_mas = x.copy()
                    x_menos = x.copy()
                    
                    x_mas[i] += self.delta
                    x_menos[i] -= self.delta

                    Hes = (self.funcion.evaluar(x_mas)-2*self.funcion.evaluar(x)+self.funcion.evaluar(x_menos))/(self.delta)**2
                    guardado[i][j] = Hes
                
                else:
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()

                    x_pp[i] += self.delta
                    x_pp[j] += self.delta
                    x_pm[i] += self.delta
                    x_pm[j] -= self.delta
                    x_mp[i] -= self.delta
                    x_mp[j] += self.delta    
                    x_mm[i] -= self.delta
                    x_mm[j] -= self.delta

                    Hes = (self.funcion.evaluar(x_pp)-self.funcion.evaluar(x_pm)-self.funcion.evaluar(x_mp)+self.funcion.evaluar(x_mm))/(4*self.delta**2)
                    guardado[i][j] = Hes 
        return guardado    
            
# Main
#ECUACION 1
ec1 = calculo(F1(), 0.1)
x = [1, 2]
gradiente = ec1.Gradiente(x)
hessiana = ec1.Hessiana(x)
print("Gradiente 1: ", gradiente)
print("Hessiana 1:", hessiana)

#ECUACION 2
ec2 = calculo(F2(), 0.1)
x = [2, 1, 3]
gradiente = ec2.Gradiente(x)
hessiana = ec2.Hessiana(x)
print("Gradiente 2: ", gradiente)
print("Hessiana 2:", hessiana)

#ECUACION 3
ec3 = calculo(F3(), 0.1)
x = [1, 1, 1, 1]
gradiente = ec3.Gradiente(x)
hessiana = ec3.Hessiana(x)
print("Gradiente 3: ", gradiente)
print("Hessiana 3:", hessiana)
# IRONEDIT:1780582827:af1cfd33b52c7bb2cf3908e9b654993e7e52869a18e9cb5a802a03ba6f1488a3
