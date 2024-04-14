import numpy as np
import matplotlib.pyplot as plt

def fonction_gaussienne_inverse(x, a=10, c=0.75, b=0):
    """
    Fonction pour générer une gaussienne inversée centrée en 0.
    """
    return a - a * np.exp(-((x-b)**2) / (2 * c**2)) + a

# Générer des valeurs x
x_values = np.linspace(-3, 3, 400)

# Calculer les valeurs y pour la fonction gaussienne inversée
y_values = fonction_gaussienne_inverse(x_values)

# Affichage du graphique
plt.figure(figsize=(8, 4))
plt.plot(x_values, y_values, label='Gaussienne inversée')
plt.title("Fonction Gaussienne 'à l'envers' centrée en 0")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
