######################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR         ENG. CIVIL / PROF (UFCAT)
# ROMES ANTÔNIO BORGES                             MATE / PROF (UFCAT)
# DONIZETTI A. DE SOUZA JÚNIOR                      ENG. CIVIL (UFCAT)
######################################################################

######################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIO. PRODES DE ALGORITMOS ESTOCÁSTICOS DE CONFIABILIDADE
######################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import pyDOE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.distributions import *

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE
def SAMPLING(SETUP):
    """
    THIS FUNCTION GENERATES RANDOM SAMPLES ACCORDING ,
    TO CHOICE SAMPLING METHOD
    
    INPUT:,
    SETUP: STOCHASTIC RANDOM VARIABLES DESCRIPTION (DICTIONARY MIXED)

    OUTPUT:
    RANDOM_SAMPLING: STOCHASTIC RANDOM SAMPLING (NP.ARRAY, FLOAT)
    """

    # START RESERVED SPACE FOR SAMPLING
    N_SAMPLING = SETUP['TOTAL SAMPLING']
    D = SETUP['TOTAL DESIGN VARIABLES']
    MODEL = SETUP['MODEL']
    RANDOM_SAMPLING = np.zeros((N_SAMPLING, D))
    if MODEL == 'MCS':
        for I_COUNT in range(D):
            # SETUP TYPE, MEAN AND STD I VARIABLE
            TYPE = SETUP['VARS'][I_COUNT][0]
            MEAN = SETUP['VARS'][I_COUNT][1]
            STD = SETUP['VARS'][I_COUNT][2]
            # NORMAL AND GAUSSIAN DISTRIBUITION
            if (TYPE == 'GAUSSIAN' or TYPE == 'NORMAL'):
                RANDOM_NUMBERS = np.random.normal(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # GUMBEL MAXIMUM DISTRIBUITION
            elif TYPE == 'GUMBEL MAX':
                RANDOM_NUMBERS = np.random.gumbel(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # LOGNORMAL DISTRIBUITION
            elif TYPE == 'LOGNORMAL':
                RANDOM_NUMBERS = np.random.lognormal(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # GAMMA DISTRIBUITION
            elif TYPE == 'GAMMA':
                RANDOM_NUMBERS = np.random.gamma(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBER                 
            # LAPLACE DISTRIBUITION
            elif TYPE == 'LAPLACE':
                RANDOM_NUMBERS = np.random.laplace(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # LOGISTIC DISTRIBUITION
            elif TYPE == 'LOGISTIC':
                RANDOM_NUMBERS = np.random.logistic(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # MULTINOMIAL DISTRIBUITION
            elif TYPE == 'MULTINOMIAL':
                RANDOM_NUMBERS = np.random.multinomial(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # MULTIVARIATE NORMAL
            elif TYPE == 'MULTIVARIATE NORMAL':
                RANDOM_NUMBERS = np.random.multivariate_normal(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # NEGATIVE BINOMIAL DISTRIBUITION
            elif TYPE == 'NEGATIVE BINOMIAL':
                RANDOM_NUMBERS = np.random.negative_binomial(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # NONCENTRAL CHISQUARE DISTRIBUITION
            elif TYPE == 'NONCENTRAL CHISQUARE':
                RANDOM_NUMBERS = np.random.noncentral_chisquare(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # UNIFORM DISTRIBUITION
            elif TYPE == 'UNIFORM':
                RANDOM_NUMBERS = np.random.uniform(MEAN, STD, N_SAMPLING)
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS  
                
    elif MODEL == 'LHS':
        DESIGN = lhs(N_SAMPLING)
        NEW_TOTAL_SAMPLING=N_SAMPLING
        NEW_ARRAY_RANDOM = []
        for I_COUNT in range(D):
            # SETUP TYPE, MEAN AND STD I VARIABLE
            TYPE = SETUP['VARS'][I_COUNT][0]
            MEAN = SETUP['VARS'][I_COUNT][1]
            STD = SETUP['VARS'][I_COUNT][2]
            
            # NORMAL AND GAUSSIAN DISTRIBUITION
            if (TYPE == 'NORMAL'):
                ARRAY_RANDOM = norm(loc=MEAN, scale=STD).ppf(DESIGN)   #MODIFICAR CONFORME MCS E TESTAR
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      NEW_ARRAY_RANDOM.append(J_AUX)
                
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]   
            # GUMBEL MAXIMUM DISTRIBUITION        
            elif TYPE == 'GUMBEL':
                ARRAY_RANDOM = gumbel_r(loc=MEAN, scale=STD).ppf(DESIGN)  #MODIFICAR CONFORME MCS E TESTAR
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      NEW_ARRAY_RANDOM.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]
            # LOGNORMAL MAXIMUM DISTRIBUITION        
            elif TYPE == 'LOGNORMAL':
                ARRAY_RANDOM = lognorm(loc=MEAN, scale=STD).ppf(DESIGN)  #MODIFICAR CONFORME MCS E TESTAR
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      NEW_ARRAY_RANDOM.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]   
            
            
             # DONIZETTI ADD OUTRAS DISTRIBUIÇÕES
 
    return RANDOM_SAMPLING
