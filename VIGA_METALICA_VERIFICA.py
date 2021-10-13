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
  LAMBDA    | Esbeltez calculada                    |  -  | Float  
  LAMBDA_R  | Esbeltez secao compacta calculada     |  -  | Float  
  LAMBDA_P  | Esbeltez secao semicompacta calculada |  -  | Float 
  Saída:
  TIPO_SEC  | Tipo de secao calculada               |  -  | String                     
  """
  if LAMBDA <= LAMBDA_P:
      TIPO_SEC = "COMPACTA"
  elif (LAMBDA > LAMBDA_P) and (LAMBDA <= LAMBDA_R):
      TIPO_SEC = "SEMI-COMPACTA"
  elif LAMBDA > LAMBDA_R:
      TIPO_SEC = "ESBELTA"
  return TIPO_SEC

def MOMENTO_MRD_ALMA(E_S, F_Y, H_W, T_W, Z, W_C, W_T, PARAMETRO_PERFIL, TIPO_PERFIL, GAMMA_A1):
    """
    Esta função classifica e verifica o momento resistente na alma de um perfil metálico
    de acordo com a NBR 8800.
    Entrada:
    E_S       | Módulo de elasticidade do aço       | kN/m² | Float
    F_Y       | Tensão de escoamento do aço         | kN/m² | Float
    H_W       | Altura da alma                      |   m   | Float
    T_W       | Largura da alma                     |   m   | Float  
    T_F       | Altura da mesa                      |   m   | Float 
    B_F       | Largura da mesa                     |   m   | Float  
    Z         | Módulo plastico da seção            |   m³  | Float
    W_C       | Módulo plastico da seção compressao | kN/m² | Float
    W_T       | Módulo plastico da seção tracao     | kN/m² | Float    
    PERFIL    | Caracteristica do perfil            |       | String
    GAMMA_A1  | Coeficiente de ponderação           |       | Float
    Saída: 
    M_RD      | Momento resistido de projeto        | kN*m  | Float
    """

    # Classificação perfil
    LAMBDA = H_W / T_W
    LAMBDA_R = 5.70 * (E_S / F_Y) ** 0.5
    if PARAMETRO_PERFIL == "DUPLA SIMETRIA":
        D = 3.76
    elif PARAMETRO_PERFIL == "MONO SIMETRIA":
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

def MOMENTO_MRD_MESA(E_S, F_Y, H_W, T_W, B_F, T_F, Z, W_C, W_T, PARAMETRO_PERFIL, TIPO_PERFIL, GAMMA_A1):
    """
    Esta função classifica e verifica o momento resistente na mesa de um perfil metálico
    de acordo com a NBR 8800.
    Entrada:
    E_S       | Módulo de elasticidade do aço       | kN/m² | Float
    F_Y       | Tensão de escoamento do aço         | kN/m² | Float
    H_W       | Altura da alma                      |   m   | Float
    T_W       | Largura da alma                     |   m   | Float  
    T_F       | Altura da mesa                      |   m   | Float 
    B_F       | Largura da mesa                     |   m   | Float  
    Z         | Módulo plastico da seção            |   m³  | Float
    W_C       | Módulo plastico da seção compressao | kN/m² | Float
    W_T       | Módulo plastico da seção tracao     | kN/m² | Float    
    PERFIL    | Caracteristica do perfil            |   -   | String
    GAMMA_A1  | Coeficiente de ponderação           |   -   | Float
    Saída: 
    M_RD      | Momento fletor resistido            | kN*m  | Float
    """

    # Classificação perfil
    LAMBDA = B_F / (2 * T_F)
    LAMBDA_P = 0.38 * (E_S / F_Y) ** 0.5 
    if TIPO_PERFIL == "SOLDADO":
        C = 0.95
        AUX_0 = 4 / (H_W / T_W) ** 0.50
        K_C = np.clip(AUX_0, 0.35, 0.76)
    elif TIPO_PERFIL == "LAMINADO":
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
    E_S       | Módulo de elasticidade do aço       | kN/m² | Float
    F_Y       | Tensão de escoamento do aço         | kN/m² | Float
    H_W       | Altura da alma                      |   m   | Float
    T_W       | Largura da alma                     |   m   | Float
    Saída:
    C_V       | Coeficiente de cisalhmento          |   -   | Float 
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
    E_S       | Módulo de elasticidade do aço       | kN/m² | Float
    F_Y       | Tensão de escoamento do aço         | kN/m² | Float
    H_W       | Altura da alma                      |   m   | Float
    T_W       | Largura da alma                     |   m   | Float
    GAMMA_A1  | Coeficiente de ponderação           |   -   | Float
    Saída:
    V_RD      | Esforco cortante resistido          |  kN   | Float
    
    """
    A_W = H_W * T_W
    C_V = CALCULO_CV(H_W, T_W, E_S, F_Y)
    V_RD = (C_V * 0.6 * F_Y * A_W) / GAMMA_A1
    return V_RD

def VERIFICACAO_VIGA_METALICA_MOMENTO_FLETOR(VIGA, ESFORCOS):
    """
    Esta função verifica o momento fletor resistente de um perfil metálico
    de acordo com a NBR 8800.
    
    Entrada:
    VIGA      | Tags dicionario  
              | 'E_S'   ==  Módulo de elasticidade do aço        | Float
              | 'F_Y'   ==  Tensão de escoamento do aço          | Float
              | 'H_W'   ==  Altura da alma                       | Float
              | 'T_W'   ==  Largura da alma                      | Float 
              | 'T_F'   ==  Altura da mesa                       | Float 
              | 'B_F'   ==  Largura da mesa                      | Float
              | 'Z'     ==  Módulo plastico da seção             | Float
              | 'W_C'   ==  Módulo plastico da seção compressao  | Float
              | 'W_T'   ==  Módulo plastico da seção tracao      | Float 
              | 'S1'    ==  Erro de modelo 1                     | Float
              | 'S2'    ==  Erro de modelo 2                     | Float    
              | 'PARAMETRO_PERFIL' ==  Caracteristica do perfil  | String
              | 'TIPO_PERFIL'      ==  Tipo de perfil analisado  | String
              | 'GAMMA_A1'         ==  Coeficiente de ponderação | Float 
    ESFORCOS  | Tags dicionario
              | 'M_SD'  == Momento fletor solicitante            | Float  
    Saída:
    R         | Momento fletor resistente                        | Float
    S         | Momento fletor solicitante                       | Float
    """

    E_S = VIGA['E_S']
    F_Y = VIGA['F_Y']
    H_W = VIGA['H_W']
    T_W = VIGA['T_W']
    B_F = VIGA['B_F']
    T_F = VIGA['T_F']
    Z = VIGA['Z']
    INERCIA = VIGA['INERCIA']
    S1 = VIGA['S1']
    S2 = VIGA['S2']
    Y_GC = (( T_F * 2) + H_W) / 2 
    W_C = INERCIA / Y_GC
    W_T = W_C
    PARAMETRO_PERFIL = VIGA['PARAMETRO_PERFIL']
    TIPO_PERFIL = VIGA['TIPO_PERFIL'] 
    GAMMA_A1 = VIGA['GAMMA_A1']
    M_SD = ESFORCOS['M_SD']

    #Resistencia momento fletor de projeto
    M_RDMESA = MOMENTO_MRD_MESA(E_S, F_Y, H_W, T_W, B_F, T_F, Z, W_C, W_T, PARAMETRO_PERFIL, TIPO_PERFIL, GAMMA_A1)
    M_RDALMA = MOMENTO_MRD_ALMA(E_S, F_Y, H_W, T_W, Z, W_C, W_T, PARAMETRO_PERFIL, TIPO_PERFIL, GAMMA_A1)
    M_RD = min(M_RDMESA, M_RDALMA)
    R = S1 * M_RD
    S = S2 * M_SD

    return(R, S)


def VERIFICACAO_VIGA_METALICA_ESFORCO_CORTANTE(VIGA, ESFORCOS):
    """
    Esta função verifica o esforco cortante de um perfil metálico
    de acordo com a NBR 8800.
    
    Entrada:
    VIGA      | Tags dicionario  
              | 'E_S'   ==  Módulo de elasticidade do aço        | Float
              | 'F_Y'   ==  Tensão de escoamento do aço          | Float
              | 'H_W'   ==  Altura da alma                       | Float
              | 'T_W'   ==  Largura da alma                      | Float 
              | 'S1'    ==  Erro de modelo 1                     | Float
              | 'S2'    ==  Erro de modelo 2                     | Float  
              | 'GAMMA_A1'  ==  Coeficiente de ponderação        | Float 
    ESFORCOS  | Tags dicionario 
              | 'V_SD'  == Esforco cortante solicitante          | Float 
    Saída:
    R         | Esforco cortante resistente                      | Float
    S         | Esforco cortante solicitante                     | Float
    """

    E_S = VIGA['E_S']
    F_Y = VIGA['F_Y']
    H_W = VIGA['H_W']
    T_W = VIGA['T_W']
    S1 = VIGA['S1']
    S2 = VIGA['S2']
    V_SD = ESFORCOS['V_SD']
    GAMMA_A1 = VIGA['GAMMA_A1']

    #Resistencia esforco cortante de projeto
    V_RD = CORTANTE_VRD(H_W, T_W, E_S, F_Y, GAMMA_A1)
    R = S1 * V_RD
    S = S2 * V_SD

    return(R, S)


def VERIFICACAO_VIGA_METALICA_DEFORMACAO(VIGA, ESFORCOS):
    """
    Esta função verifica o deflexao maxima de um perfil metálico
    de acordo com a NBR 8800.
    
    Entrada:
    VIGA      | Tags dicionario  
              | 'L_MAX' == Largura do elemento                   | Float 
              | 'S1'    ==  Erro de modelo 1                     | Float
              | 'S2'    ==  Erro de modelo 2                     | Float  
    ESFORCOS  | Tags dicionario 
              | 'D_SD'  == Deflexao solicitante                  | Float 
    Saída:
    R         | Esforco cortante resistente                      | Float
    S         | Esforco cortante solicitante                     | Float
    """

    D_SD = ESFORCOS['D_SD']
    S1 = VIGA['S1']
    S2 = VIGA['S2']
    L_MAX = ESFORCOS['L_MAX']
    D_MAX = L_MAX / 350

    R = S1 * D_MAX
    S = S2 * D_SD / 100

    return(R, S)

def INERCIA_CALCULO(B_F, T_F, H_W, T_W):

    CG_A1 = B_F * T_F
    CG_Y1 =  T_F/2
    CG_PT1 = CG_A1 * CG_Y1

    CG_A2 = H_W * T_W 
    CG_Y2 = (T_F)+(H_W/2)
    CG_PT2 = CG_A2 * CG_Y2

    CG_A3 = B_F * T_F
    CG_Y3 = (T_F/2)+(T_F+H_W)
    CG_PT3 = CG_A3 * CG_Y3

    CG = (CG_PT1+CG_PT2+CG_PT3)/(CG_A1+CG_A2+CG_A3)

    INERCIA1_PT1 = (B_F * (T_F**3))/12
    INERCIA1_PT2 = (CG - CG_Y1) ** 2
    INERCIA1_PT3 = B_F * T_F
    INERCIA1 = INERCIA1_PT1 + INERCIA1_PT3 * INERCIA1_PT2

    INERCIA2_PT1 = (T_W * (H_W **3))/12
    INERCIA2_PT2 = (CG - CG_Y2) ** 2
    INERCIA2_PT3 = H_W * T_W 
    INERCIA2 = INERCIA2_PT1 + INERCIA2_PT3 * INERCIA2_PT2

    INERCIA3_PT1 = (B_F * (T_F**3))/12
    INERCIA3_PT2 = (CG - CG_Y3) ** 2
    INERCIA3_PT3 = B_F * T_F
    INERCIA3 = INERCIA3_PT1 + INERCIA3_PT3 * INERCIA3_PT2
    INERCIA_TOTAL = (INERCIA1 + INERCIA2 + INERCIA3) * 0.000000000001 #mm4 para m4

    return (INERCIA_TOTAL)

def MODULO_PLASTICO(B_F, T_F, H_W, T_W):
    H = (H_W)+(2*T_F)
    PT1 = B_F * T_F * (H - T_F)
    PT2 = (T_W/4)*(H - 2 * T_F)**2

    Z = (PT1 + PT2) * 0.000000001 #mm3 para m3

    return(Z)