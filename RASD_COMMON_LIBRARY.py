######################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR         ENG. CIVIL / PROF (UFCAT)
# ROMES ANTÔNIO BORGES                             MATE / PROF (UFCAT)
# DONIZETTI APARECIDO DE S. JÚNIOR                  ENG. CIVIL (UFCAT)
######################################################################

######################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIO. PRODES DE ALGORITMOS ESTOCÁSTICOS DE CONFIABILIDADE
######################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
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
                RANDOM_SAMPLING[:, I_COUNT] = RANDOM_NUMBERS
            # DONIZETTI ADD OUTRAS DISTRIBUIÇÕES
            ####
            ####
            ####
            ####
            ####
    elif MODEL == 'LHS':
        DESIGN = lhs(TOTAL_SAMPLING)
        for I_COUNT in range(D):
            # SETUP TYPE, MEAN AND STD I VARIABLE
            TYPE = SETUP['VARS'][I_COUNT][0]
            MEAN = SETUP['VARS'][I_COUNT][1]
            STD = SETUP['VARS'][I_COUNT][2]
            # NORMAL AND GAUSSIAN DISTRIBUITION
            if (TYPE == 'NORMAL'):
                RANDOM_NUMBERS=[]
                RANDOM_NUMBERS = norm(loc=MEAN, scale=STD).ppf(DESIGN)
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      RANDOM_NUMBERS.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]
                    
            if (TYPE == 'GUMBEL'):
                RANDOM_NUMBERS=[]
                RANDOM_NUMBERS = gumbel_r(loc=MEAN, scale=STD).ppf(DESIGN)
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      RANDOM_NUMBERS.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]
                    
            if (TYPE == 'LOGNORMAL'):
                RANDOM_NUMBERS=[]
                RANDOM_NUMBERS = lognorm(loc=MEAN, scale=STD).ppf(DESIGN)
                for I_AUX in ARRAY_RANDOM:
                    for J_AUX in I_AUX:
                      RANDOM_NUMBERS.append(J_AUX)
                for J_COUNT in range (NEW_TOTAL_SAMPLING):
                    RANDOM_SAMPLING[J_COUNT,I_COUNT]=NEW_ARRAY_RANDOM[J_COUNT]   
                    
             # DONIZETTI ADD OUTRAS DISTRIBUIÇÕES
 
    return RANDOM_SAMPLING
