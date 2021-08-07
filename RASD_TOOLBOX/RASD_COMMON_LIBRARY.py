"""
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::..::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::...::...:::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::..:!%@&&@%!:..::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::..:!%@###@@###@%!:..:::::::::::::::::::::::::::::::
::::::::::::::::::::::::::..:!%@##&$*::::*$&##@%!:..::::::::::::::::::::::::::::
::::::::::::::::::::::::::!%&##&$*::*$&&$*::*$&##&%!::::::::::::::::::::::::::::
::::::::::::::::::::::::.$##&$*::*@&##@@&@*.:::!$&##$.::::::::::::::::::::::::::
::::::::::::::::::::::::.@#&!.:@####%:..::!$&&@*.:&#@:::::::::::::::::::::::::::
::::::::::::::::::::::::.@#&:.:!%&##&$**$&##&&##::&#@:::::::::::::::::::::::::::
::::::::::::::::::::::::.@#&::$%::!%@####&%!:$#&::&#$:::::::::::::::::::::::::::
::::::::::::::::::::::::.@#&::##@...:!%%!:...$#&!:*:::::::::::::::::::::::::::::
::::::::::::::::::::::::.@#&::&#@.:::....:::.$#&!.*$$:::::::::::::::::::::::::::
::::::::::::::::::::::::.@#&::##&!:..::::..:!@##!:&#@:::::::::::::::::::::::::::
::::::::::::::::::::::::.@#&::$&##@%!:..:!%@##&$::&#@:::::::::::::::::::::::::::
::::::::::::::::::::::::.@##@*::*$&##@$%@##&$*::*@##@:::::::::::::::::::::::::::
::::::::::::::::::::::::::%@##&@*::*$&##&$*::*$&##@%!:::::::::::::::::::::::::::
:::::::::::::::::::::::::..:!%@###@*::!!::*@###@%!:..:::::::::::::::::::::::::::
::::::::::::::::::::::::::::...:*@###@%%@###@%!:..::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::...:*$&##&@*:...:::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::...:**:...::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::...:::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::........::.......::........:........:::::::::::::::::::::::
:::::::::::::::::::::%$$$$$$*.:%$$$$$%::*$$$$$$*:%$$$$$$*:::::::::::::::::::::::
:::::::::::::::::::::&&****@#:%#%***$#*!#$*****!:&&****@&:::::::::::::::::::::::
:::::::::::::::::::::&&@@@$@$.%#@@@@@#*:@@$@@@@*:&@....$&:::::::::::::::::::::::
:::::::::::::::::::::&@:%&@*!.$#!:::*#*:!*****#@.&@!!!!@#:::::::::::::::::::::::
:::::::::::::::::::::$%..:$@$:*@:...!@!:@@@@@@@!:$@@@@@@%:::::::::::::::::::::::
:::::::::::::::::::::..::....::.:::::.:::.......:.....:..:::::::::::::::::::::::
"""
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
from scipy.stats.distributions import *

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE
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
                RANDOM_NUMBERS = np.random.normal(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Gumbel maximum
            elif TYPE == 'GUMBEL MAX':
                RANDOM_NUMBERS = np.random.gumbel(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Lognormal
            elif TYPE == 'LOGNORMAL':
                RANDOM_NUMBERS = np.random.lognormal(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Gamma
            elif TYPE == 'GAMMA':
                RANDOM_NUMBERS = np.random.gamma(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBER                 
            # Laplace
            elif TYPE == 'LAPLACE':
                RANDOM_NUMBERS = np.random.laplace(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Logistic
            elif TYPE == 'LOGISTIC':
                RANDOM_NUMBERS = np.random.logistic(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Multinomial
            elif TYPE == 'MULTINOMIAL':
                RANDOM_NUMBERS = np.random.multinomial(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Multivariate normal
            elif TYPE == 'MULTIVARIATE NORMAL':
                RANDOM_NUMBERS = np.random.multivariate_normal(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Negative binomial
            elif TYPE == 'NEGATIVE BINOMIAL':
                RANDOM_NUMBERS = np.random.negative_binomial(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Noncentral chisquare
            elif TYPE == 'NONCENTRAL CHISQUARE':
                RANDOM_NUMBERS = np.random.noncentral_chisquare(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # Uniform
            elif TYPE == 'UNIFORM':
                RANDOM_NUMBERS = np.random.uniform(MEAN, STD, N_POP)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS  
    # Latin Hyper Cube Sampling            
    elif MODEL == 'LHS':
        DESIGN = lhs(N_POP)
        NEW_TOTAL_SAMPLING=N_POP
        NEW_ARRAY_RANDOM = []
        for I_COUNT in range(D):
            # Type of distribution, mean and standard deviation
            TYPE = VARS[I_COUNT][0]
            MEAN = VARS[I_COUNT][1]
            STD = VARS[I_COUNT][2]  
            # Normal or Gaussian
            if (TYPE == 'NORMAL'):
                ARRAY_RANDOM = norm(loc=MEAN, scale=STD).ppf(DESIGN)   
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      NEW_ARRAY_RANDOM.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]   
            # Gumbel maximum       
            elif TYPE == 'GUMBEL':
                ARRAY_RANDOM = gumbel_r(loc=MEAN, scale=STD).ppf(DESIGN)  
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      NEW_ARRAY_RANDOM.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]
            # Lognormal maximum     
            elif TYPE == 'LOGNORMAL':
                ARRAY_RANDOM = lognorm(loc=MEAN, scale=STD).ppf(DESIGN)  
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      NEW_ARRAY_RANDOM.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]   
    return RANDOM_SAMPLING


# DISTRIBUITIONS
# https://docs.scipy.org/doc/numpy-1.9.3/reference/routines.random.html
# https://docs.scipy.org/doc/scipy/reference/stats.html 
