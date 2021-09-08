################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# DONIZETTI SOUZA JUNIOR,                                     ENG. CIVIL (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA DE VERIFICAÇÃO DE ESTADO LIMITE EM VIGAS METÁLICAS DESENVOLVIDA 
# PELO GRUPO DE PESQUISAS E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS PYTHON
import numpy as np

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE

def DEFINICAO_SECAO(LAMBDA, LAMBDA_R, LAMBDA_P):
  """
  Está função define se uma seção metálica é compacta, semi-compacta ou esbelta.

  Entrada:
  LAMBDA    | Esbeltez calculada para o perfil    |      | Float  
  LAMBDA_R  |                                     |      | Float  
  LAMBDA_P  |                                     |      | Float 

  Saída:
  TIPO_SEC  |                                     |      | String                     
  """
  if LAMBDA <= LAMBDA_P:
      TIPO_SEC = "COMPACTA"
  elif (LAMBDA > LAMBDA_P) and (LAMBDA <= LAMBDA_R):
      TIPO_SEC = "SEMI-COMPACTA"
  elif LAMBDA > LAMBDA_R:
      TIPO_SEC = "ESBELTA"
  return TIPO_SEC

def MOMENTO_MRD_ALMA(E_S, F_Y, H_W, T_W, Z, W_C, W_T, PERFIL, GAMMA_A1):
    """
    Esta função classifica e verifica o momento resistente na alma de um perfil metálico
    de acordo com a NBR 8800.

    Entrada:
    E_S       | Módulo de elasticidade do aço       | kN/m²  | Float
    F_Y       | Tensão de escoamento do aço         | kN/m²  | Float
    H_W       |
    T_W       |
    Z         |
    W_C       |
    W_T       |
    PERFIL    |
    GAMMA_A1  |

    Saída: 
    M_RD      |
    """
    # Classificação perfil
    LAMBDA = H_W / T_W
    LAMBDA_R = 5.70 * (E_S / F_Y) ** 0.5
    if PERFIL == "DUPLA SIMETRIA":
        D = 3.76
    elif PERFIL == "MONO SIMETRIA":
        pass # depois tem que implementar o monosimentrico
    LAMBDA_P = D * (E_S / F_Y) ** 0.5
    if LAMBDA_P >= LAMBDA_R: 
        print('SEÇÃO COM λp > λr')
        # return Aqui tem que ver como vamos fazer para o código encerrar aqui
    TIPO_SEC = DEFINICAO_SECAO(LAMBDA, LAMBDA_R, LAMBDA_P)
    # Momento resistente
    if TIPO_SEC == "COMPACTA":
        M_P = Z * F_Y
        M_RD = M_P / GAMMA_A1
    elif TIPO_SEC == "SEMI-COMPACTA":
        M_P = Z * F_Y
        W = min(W_C, W_T)
        M_R = W * F_Y 
        AUX_0 = (LAMBDA - LAMBDA_P) / (LAMBDA_R - LAMBDA_P)
        M_RD = (M_P - AUX_0 * (M_P - M_R)) / GAMMA_A1
    elif TIPO_SEC == "ESBELTA":      
        pass
    # Momento máximo resistente
    W = min(W_C, W_T)
    M_RDMAX = 1.50 * W * F_Y / GAMMA_A1
    if M_RD > M_RDMAX:
        M_RD = M_RDMAX
    return M_RD

def MOMENTO_MRD_MESA(E_S, F_Y, H_W, T_W, B_F, T_F, Z, W_C, W_T, PERFIL, GAMMA_A1):
    """
    Esta função classifica e verifica o momento resistente na mesa de um perfil metálico
    de acordo com a NBR 8800.

    Entrada:
    E_S       | Módulo de elasticidade do aço       | kN/m²  | Float
    F_Y       | Tensão de escoamento do aço         | kN/m²  | Float
    H_W       |
    T_W       |   
    B_F       |
    T_F       |
    Z         |
    W         |
    PERFIL    |
    GAMMA_A1  |

    Saída: 
    M_RD      |
    """
    # Classificação perfil
    LAMBDA = B_F / (2 * T_F)
    LAMBDA_P = 0.38 * (E_S / F_Y) ** 0.5 
    if PERFIL == "SOLDADO":
        C = 0.95
        AUX_0 = 4 / (H_W / T_W) ** 0.50
        K_C = np.clip(AUX_0, 0.35, 0.76)
    elif PERFIL == "LAMINADO":
        C = 0.83
        K_C = 1.00
    AUX_1 = 0.70 * F_Y / K_C
    LAMBDA_R = C * (E_S / AUX_1) ** 0.5  
    TIPO_SEC = DEFINICAO_SECAO(LAMBDA, LAMBDA_R, LAMBDA_P)
    # Momento resistente
    if TIPO_SEC == "COMPACTA":
        M_P = Z * F_Y
        M_RD = M_P / GAMMA_A1
    elif TIPO_SEC == "SEMI-COMPACTA":
        M_P = Z * F_Y
        SIGMA_R = 0.30 * F_Y
        M_R = W_C * (F_Y - SIGMA_R) 
        M_RMAX = W_T * F_Y
        if M_R > M_RMAX:
            M_R = M_RMAX
        AUX_2 = (LAMBDA - LAMBDA_P) / (LAMBDA_R - LAMBDA_P)
        M_RD = (M_P - AUX_2 * (M_P - M_R)) / GAMMA_A1
    elif TIPO_SEC == "ESBELTA":      
        if PERFIL == "SOLDADO":
            M_N = (0.90 * E_S * K_C * W_C) / LAMDA ** 2 
        elif PERFIL == "LAMINADO":
            M_N = (0.69 * E_S * W_C) / LAMDA ** 2
        M_RD = M_N / GAMMA_A1 
    W = min(W_C, W_T)   
    # Momento máximo resistente
    M_RDMAX = 1.50 * W * F_Y / GAMMA_A1
    if M_RD > M_RDMAX:
        M_RD = M_RDMAX
    return M_RD
 
def CALCULO_CV(H_W, T_W, E_S, F_Y):
    """
    Esta função determina o coeficiente de redução do cisalhamento resistente Cv.

    Entrada:

    Saída:

    """
    LAMBDA = H_W / T_W
    LAMBDA_P = 2.46 * (E_S / F_Y) ** 0.5
    LAMBDA_R = 3.06 * (E_S / F_Y) ** 0.5
    if LAMBDA <= LAMBDA_P:
        C_V = 1
    elif (LAMBDA_P < LAMBDA) and (LAMBDA <= LAMBDA_R):
        C_V = (2.46 / LAMBDA) * (E_S / F_Y) ** 0.5
    elif LAMBDA > LAMBDA_R:
        C_V = (7.5 * E_S) / (F_Y * LAMBDA ** 2)
    return C_V

def CORTANTE_VRD(H_W, T_W, E_S, F_Y, GAMMA_A1):
    """
    Esta função determina o cortante de cálculo para seções metálicas segundo a NBR 8800.
    
    Entrada:

    Saída:
    
    """
    A_W = H_W * T_W
    C_V = CALCULO_CV(H_W, T_W, E_S, F_Y)
    V_RD = (C_V * 0.6 * F_Y * AW) / GAMMA_A1
    return V_RD

def VERIFICA_VIGA_METALICA(VIGA):

    G = [];  R = [];   S = []
    #Demandas de projeto
    M_SDMAX = VIGA['M_SDMAX']
    V_SDMAX = VIGA['V_SDMAX']
    DELTA_SDMAX = VIGA['DELTA_SDMAX']

    #Resistencia momento fletor de projeto
    M_RDMESA = MOMENTO_MRD_MESA(E_S, F_Y, H_W, T_W, B_F, T_F, Z, W_C, W_T, PERFIL, GAMMA_A1)
    M_RDALMA = MOMENTO_MRD_ALMA(E_S, F_Y, H_W, T_W, Z, W_C, W_T, PERFIL, GAMMA_A1)
    M_RD = min(M_RDMESA, M_RDALMA)

    R.append(M_RD)
    S.append(M_SDMAX)
    G.append(-M_RD + M_SDMAX)

    #Resistencia esforco cortante de projeto
    
