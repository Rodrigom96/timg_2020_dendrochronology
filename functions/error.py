import numpy as np
def relative_error (x1, x2):
    """
    Entradas:
        x1: valor real
        x2: valor experimental
    Salidas:
        Error relativo en porcentaje
    """
    return np.abs(x1 - x2)/x1*100