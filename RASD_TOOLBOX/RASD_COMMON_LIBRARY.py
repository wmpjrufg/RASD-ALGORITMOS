#  /$$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$$        /$$$$$$$$ /$$$$$$   /$$$$$$  /$$       /$$$$$$$   /$$$$$$  /$$   /$$
# | $$__  $$ /$$__  $$ /$$__  $$| $$__  $$      |__  $$__//$$__  $$ /$$__  $$| $$      | $$__  $$ /$$__  $$| $$  / $$
# | $$  \ $$| $$  \ $$| $$  \__/| $$  \ $$         | $$  | $$  \ $$| $$  \ $$| $$      | $$  \ $$| $$  \ $$|  $$/ $$/
# | $$$$$$$/| $$$$$$$$|  $$$$$$ | $$  | $$         | $$  | $$  | $$| $$  | $$| $$      | $$$$$$$ | $$  | $$ \  $$$$/ 
# | $$__  $$| $$__  $$ \____  $$| $$  | $$         | $$  | $$  | $$| $$  | $$| $$      | $$__  $$| $$  | $$  >$$  $$ 
# | $$  \ $$| $$  | $$ /$$  \ $$| $$  | $$         | $$  | $$  | $$| $$  | $$| $$      | $$  \ $$| $$  | $$ /$$/\  $$
# | $$  | $$| $$  | $$|  $$$$$$/| $$$$$$$/         | $$  |  $$$$$$/|  $$$$$$/| $$$$$$$$| $$$$$$$/|  $$$$$$/| $$  \ $$
# |__/  |__/|__/  |__/ \______/ |_______/          |__/   \______/  \______/ |________/|_______/  \______/ |__/  |__/

################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR                   ENG. CIVIL / PROF (UFCAT)
# ROMES ANTÔNIO BORGES                                       MAT. / PROF (UFCAT)
# DONIZETTI A. DE SOUZA JÚNIOR                                ENG. CIVIL (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA RASD DE ALGORITMOS ESTOCÁSTICOS DE CONFIABILIDADE DESENVOLVIDOS
# PELO GRUPO DE PESQUISA DE ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
from pyDOE import *
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from scipy.stats.distributions import gumbel_r
from scipy.stats.distributions import gumbel_l
from scipy.stats.distributions import lognorm
from scipy.stats.distributions import uniform
from scipy.integrate import quad

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE

# CRIAÇÃO DAS AMOSTRAS VIA MÉTODOS ESTOCÁSTICOS 
def SAMPLING(N_POP, D, MODEL, VARS):
    """
    This function generates random samples according to chosen sampling method.
    
    Input:
    N_POP            |  Total of samplings                      | Integer
    D                |  Number of variables                     | Integer  
    MODEL            |  Algorithm setup                         | String
                     |      'MCS': Monte Carlo Sampling         |
                     |      'LHS': Latim Hypercube Sampling     | 
    VARS             |  Description of variables                | Py list[D]
                     |      Example:                            |
                     |      V_1 = ['NORMAL', 500, 100]          |
                     |      V_2 = ['NORMAL', 1000, 1000]        |
                     |      VARS = [V_1, V_2]                   |

    Output:
    RANDOM_SAMPLING  |  Samples                                 | Py Numpy array[N_POP x D]
    """
    # Creating variables
    RANDOM_SAMPLING = np.zeros((N_POP, D))
    # Monte Carlo sampling
    if MODEL == 'MCS':
        for I_COUNT in range(D):
            # Type of distribution, mean and standard deviation
            TYPE = VARS[I_COUNT][0]
            MEAN = VARS[I_COUNT][1]
            STD = VARS[I_COUNT][2]
            # Normal or Gaussian
            if (TYPE == 'GAUSSIAN' or TYPE == 'NORMAL'):
                RANDOM_SAMPLING[:, I_COUNT] = norm.rvs(loc = MEAN, scale = STD, size = N_POP, random_state = None)
            # Gumbel right or Gumbel maximum
            elif TYPE == 'GUMBEL MAX':
                RANDOM_SAMPLING[:, I_COUNT] = gumbel_r.rvs(loc = MEAN, scale = STD, size = N_POP, random_state = None)
            # Gumbel left or Gumbel minimum
            elif TYPE == 'GUMBEL MIN':
                RANDOM_SAMPLING[:, I_COUNT] = gumbel_l.rvs(loc = MEAN, scale = STD, size = N_POP, random_state = None)
            # Lognormal
            elif TYPE == 'LOGNORMAL':
                RANDOM_SAMPLING[:, I_COUNT] = lognormal.rvs(loc = MEAN, scale = STD, size = N_POP, random_state = None)
            # Uniform
            elif TYPE == 'UNIFORM':
                RANDOM_SAMPLING[:, I_COUNT] = uniform.rvs(loc = MEAN, scale = STD, size = N_POP, random_state = None)
    # Latin Hyper Cube Sampling            
    elif MODEL == 'LHS':
        RANDOM_SAMPLING = lhs(D, criterion = 'center', samples = N_POP)
        for I_COUNT in range(D):
            # Type of distribution, mean and standard deviation
            TYPE = VARS[I_COUNT][0]
            MEAN = VARS[I_COUNT][1]
            STD = VARS[I_COUNT][2]  
            # Normal or Gaussian
            if (TYPE == 'GAUSSIAN' or TYPE == 'NORMAL'):
                RANDOM_SAMPLING[:, I_COUNT] = norm(loc = MEAN, scale = STD).ppf(RANDOM_SAMPLING[:, I_COUNT])
            # Gumbel right or Gumbel maximum       
            elif TYPE == 'GUMBEL MAX':
                RANDOM_SAMPLING[:, I_COUNT] = gumbel_r(loc = MEAN, scale = STD).ppf(RANDOM_SAMPLING[:, I_COUNT])
            # Gumbel left or Gumbel minimum
            elif TYPE == 'GUMBEL MIN':
                RANDOM_SAMPLING[:, I_COUNT] = gumbel_l(loc = MEAN, scale = STD).ppf(RANDOM_SAMPLING[:, I_COUNT])
            # Lognormal
            elif TYPE == 'LOGNORMAL':
                RANDOM_SAMPLING[:, I_COUNT] = lognormal(loc = MEAN, scale = STD).ppf(RANDOM_SAMPLING[:, I_COUNT])
            # Uniform
            elif TYPE == 'UNIFORM':
                RANDOM_SAMPLING[:, I_COUNT] = uniform(loc = MEAN, scale = STD).ppf(RANDOM_SAMPLING[:, I_COUNT])

    return RANDOM_SAMPLING

def PROBABILITY_OF_FAILURE_AUX(BETA_INITIAL):
  def INTEGRAND(X):
      return  ((1/(2*3.14159265359)**0.5)*2.71828182846**(-(X**2/2)))
  ans, ERR = quad(INTEGRAND, BETA_INITIAL, np.inf)
  return(ans)


def PROBABILITY_OF_FAILURE():
    SPACE_NUM = np.linspace(0,5,100000)
    DF_BETA = []
    DF_PF = []

    HEADER = ['PF','BETA']

    for i in SPACE_NUM:
      DF_BETA.append(i)
      DF_PF.append(PROBABILITY_OF_FAILURE_AUX(i))  
    BETA_DF = pd.DataFrame(np.column_stack([DF_PF, DF_BETA]), columns=['PF','BETA'])

    return(BETA_DF)



#   /$$$$$$  /$$$$$$$  /$$$$$$$$ /$$$$$$$$       /$$$$$$$$ /$$$$$$$$  /$$$$$$  /$$   /$$ /$$   /$$  /$$$$$$  /$$        /$$$$$$   /$$$$$$  /$$$$$$ /$$$$$$$$  /$$$$$$ 
#  /$$__  $$| $$__  $$| $$_____/| $$_____/      |__  $$__/| $$_____/ /$$__  $$| $$  | $$| $$$ | $$ /$$__  $$| $$       /$$__  $$ /$$__  $$|_  $$_/| $$_____/ /$$__  $$
# | $$  \__/| $$  \ $$| $$      | $$               | $$   | $$      | $$  \__/| $$  | $$| $$$$| $$| $$  \ $$| $$      | $$  \ $$| $$  \__/  | $$  | $$      | $$  \__/
# | $$ /$$$$| $$$$$$$/| $$$$$   | $$$$$            | $$   | $$$$$   | $$      | $$$$$$$$| $$ $$ $$| $$  | $$| $$      | $$  | $$| $$ /$$$$  | $$  | $$$$$   |  $$$$$$ 
# | $$|_  $$| $$____/ | $$__/   | $$__/            | $$   | $$__/   | $$      | $$__  $$| $$  $$$$| $$  | $$| $$      | $$  | $$| $$|_  $$  | $$  | $$__/    \____  $$
# | $$  \ $$| $$      | $$      | $$               | $$   | $$      | $$    $$| $$  | $$| $$\  $$$| $$  | $$| $$      | $$  | $$| $$  \ $$  | $$  | $$       /$$  \ $$
# |  $$$$$$/| $$      | $$$$$$$$| $$$$$$$$         | $$   | $$$$$$$$|  $$$$$$/| $$  | $$| $$ \  $$|  $$$$$$/| $$$$$$$$|  $$$$$$/|  $$$$$$/ /$$$$$$| $$$$$$$$|  $$$$$$/
#  \______/ |__/      |________/|________/         |__/   |________/ \______/ |__/  |__/|__/  \__/ \______/ |________/ \______/  \______/ |______/|________/ \______/ 