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
# BIBLIOTECA DE ALGORITMOS ESTOCÁSTICOS PARA ANÁLISE DE CONFIABILIDADE ESTRUTU-
# RAL DESENVOLVIDA PELO GRUPO DE PESQUISAS E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE
import RASD_TOOLBOX.RASD_COMMON_LIBRARY as RASD_CL

def RASD_STOCHASTIC(SETUP, OF_FUNCTION):
    """
    This function creates the samples and evaluates the limit state functions.
    
    Input:
    SETUP            |  Random variables description             | Py dictionary
                     |  # Dictionary tags                        | 
                     |  'N_REP': Total of repetitions            | Integer
                     |  'POP': Total of samplings in             | Py list[D]
                     |      I repetition                         | 
                     |      Example:                             |
                     |      POP = [10, 100, 300]                 |                     
                     |  'N_G': Total of State                    | Integer
                     |      Limit functions                      |
                     |  'D': Number of variables                 | Integer
                     |  'VARS': Description of variables         | Py list[D]
                     |      Example:                             |
                     |      V_1 = ['NORMAL', 500, 100]           |
                     |      V_2 = ['NORMAL', 1000, 1000]         |
                     |      VARS = [V_1, V_2]                    |
                     |  'MODEL': Algorithm setup                 | String
                     |      'MCS': Monte Carlo Sampling          |
                     |      'LHS': Latim Hypercube Sampling      |
    OF_FUNCTION      |  External def user input this function in | Py function
                     |      arguments                            |

    Output:
    RESULTS          | All Results                               | Py dictionary
                     |     'TOTAL RESULTS':                      | Py dictionary
                     |     Sampling, Resistance, Demand, Limit   |
                     |         State Function, I count           |
                     |     'NUMBER OF FAILURES': Number of       | Py list[N_G]
                     |         failure                           | 
                     |     'PROBABILITY OF FAILURE': failure     | Py list[N_G]
                     |         probability                       | 
    """
    # Initial setup
    N_REP = SETUP['N_REP']
    POP = SETUP['POP']
    N_G = SETUP['N_G']
    D = SETUP['D']
    MODEL = SETUP['MODEL']
    VARS = SETUP['VARS']
    RESULTS = []
    for J_COUNT, N_POP in enumerate(POP):
        RESULTS_X = np.zeros((N_POP, D))
        RESULTS_R = np.zeros((N_POP, N_G))
        RESULTS_S = np.zeros((N_POP, N_G))
        RESULTS_G = np.zeros((N_POP, N_G))
        RESULTS_I = np.zeros((N_POP, N_G))
        # Creating samples   
        DATASET_X = RASD_CL.SAMPLING(N_POP, D, MODEL, VARS)
        # Evaluates Limit State functions
        for I_COUNT in range(N_POP):
            # I sample 
            RESULTS_X[I_COUNT, :] = DATASET_X[I_COUNT, :]
            SAMPLE = DATASET_X[I_COUNT, :]
            # Limit State function
            [R, S, G] = OF_FUNCTION(SAMPLE)
            # Failure or not failure - I sample 
            for K_COUNT in range(N_G):
                # Resistance
                RESULTS_R[I_COUNT, K_COUNT] = R[K_COUNT]
                # Demand
                RESULTS_S[I_COUNT, K_COUNT] = S[K_COUNT]
                # Limit State function
                RESULTS_G[I_COUNT, K_COUNT] = G[K_COUNT]
                # Failure check
                if G[K_COUNT] <= 0: 
                    I = 0
                    RESULTS_I[I_COUNT, K_COUNT] = int(I)
                elif G[K_COUNT] > 0: 
                    I = 1
                    RESULTS_I[I_COUNT, K_COUNT] = int(I) 
        # Storage all results
        AUX = np.hstack((RESULTS_X, RESULTS_R, RESULTS_S, RESULTS_G, RESULTS_I))
        RESULTS_RASD = pd.DataFrame(AUX)          
        # Rename columns in dataframe 
        COLUMNS_NAMES = []
        P_F = []
        N_F = []
        for L_COUNT in range(D):
            COLUMNS_NAMES.append('X_' + str(L_COUNT))
        for L_COUNT in range(N_G):
            COLUMNS_NAMES.append('R_' + str(L_COUNT))    
        for L_COUNT in range(N_G):
            COLUMNS_NAMES.append('S_' + str(L_COUNT)) 
        for L_COUNT in range(N_G):
            COLUMNS_NAMES.append('G_' + str(L_COUNT))
        for L_COUNT in range(N_G):
            COLUMNS_NAMES.append('I_' + str(L_COUNT))
        RESULTS_RASD.columns = COLUMNS_NAMES
        # Failure probability
        for L_COUNT in range(N_G):
            INDEX = 'I_' + str(L_COUNT)
            N_FAILURE = RESULTS_RASD[INDEX].sum()
            N_F.append(N_FAILURE)
            P_FVALUE = N_FAILURE / N_POP
            P_F.append(P_FVALUE)
        # Save results
        RESULTS_REP = {'TOTAL RESULTS': RESULTS_RASD, 'NUMBER OF FAILURES': N_F, 'PROBABILITY OF FAILURE': P_F}
        RESULTS.append(RESULTS_REP)
        NAME = 'RASD_' + MODEL + '_REP_' + str(J_COUNT) + '_SAMPLES_' + str(N_POP) + '_' + str(datetime.now().strftime('%Y%m%d %H%M%S')) + '.txt'
        HEADER_NAMES =  ';'.join(COLUMNS_NAMES)
        np.savetxt(NAME, RESULTS_RASD, fmt = '%d', delimiter = ';' , header = HEADER_NAMES)
    return RESULTS


