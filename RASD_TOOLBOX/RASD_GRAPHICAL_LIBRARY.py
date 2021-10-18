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
# BIBLIOTECA GRÁFICA PARA ANÁLISE DE CONFIABILIDADE ESTRUTURAL DESENVOLVIDA 
# PELO GRUPO DE PESQUISAS E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE 

# CONVERTE SI PARA POLEGADAS NO TAMANHO DO GRÁFICO
def CONVERT_SI_TO_INCHES(WIDTH, HEIGHT):
    """ 
    This function convert figure size meters to inches.
    
    Input:
    WIDTH    |  Figure width in SI units       | Float
    HEIGHT   |  Figure height in SI units      | Float
    
    Output:
    WIDTH    |  Figure width in INCHES units   | Float
    HEIGHT   |  Figure height in INCHES units  | Float
    """
    WIDTH = WIDTH / 0.0254
    HEIGHT = HEIGHT / 0.0254
    return WIDTH, HEIGHT

# SALVA A FIGURA
def SAVE_GRAPHIC(NAME, EXT, DPI):
    """ 
    This function save graphics on a specific path extensions options.

    - 'svg'
    - 'png'
    - 'eps'
    - 'pdf'

    Input: 
    NAME  | Path + name figure               | String
    EXT   | File extension                   | String
    DPI   | The resolution in dots per inch  | Integer
    
    Output:
    N/A
    """
    plt.savefig(NAME + EXT, dpi = DPI, bbox_inches = 'tight', transparent = True)


# PLOTAGEM 1
def RASD_PLOT_1(DATASET, PLOT_SETUP):
    """
    This function shows a boxplot and histograms in a single chart.
    
    Input: 
    DATASET     | Results from a RASD Toolboox                              | Py dataframe or Py Numpy array[N_POP x 1]
                |    Dictionary tags                                        |
                |    'DATA'          == Complete data                       | Py Numpy array[N_POP x 1]
                |    'COLUMN'        == Dataframe column                    | String
    PLOT_SETUP  | Contains specifications of each model of chart            | Py dictionary
                |    Dictionary tags                                        |
                |    'NAME'          == Filename output file                | String 
                |    'WIDTH'         == Width figure                        | Float
                |    'HEIGHT         == Height figure                       | Float
                |    'X AXIS SIZE'   == X axis size                         | Float
                |    'Y AXIS SIZE'   == Y axis size                         | Float
                |    'AXISES COLOR'  == Axis color                          | String
                |    'X AXIS LABEL'  == X label name                        | String
                |    'LABELS SIZE'   == Labels size                         | Float
                |    'LABELS COLOR'  == Labels color                        | String
                |    'CHART COLOR'   == Boxplot and histogram color         | String
                |    'BINS'          == Range representing the width of     | Float
                |                       a single bar                        | 
                |    'KDE'           == Smooth of the random distribution   | Boolean      
                |    'DPI'           == Dots Per Inch - Image quality       | Integer   
                |    'EXTENSION'     == Extension output file               | String ('.svg, '.png', '.eps' or '.pdf')
    
    Output:
    N/A
    """
    # Setup
    NAME = PLOT_SETUP['NAME']
    W = PLOT_SETUP['WIDTH']
    H = PLOT_SETUP['HEIGHT']
    X_AXIS_LABEL = PLOT_SETUP['X AXIS LABEL']
    X_AXIS_SIZE = PLOT_SETUP['X AXIS SIZE']
    Y_AXIS_SIZE = PLOT_SETUP['Y AXIS SIZE']
    AXISES_COLOR = PLOT_SETUP['AXISES COLOR']
    LABELS_SIZE = PLOT_SETUP['LABELS SIZE']     
    LABELS_COLOR = PLOT_SETUP['LABELS COLOR']
    CHART_COLOR = PLOT_SETUP['CHART COLOR']
    BINS = PLOT_SETUP['BINS']
    KDE = PLOT_SETUP['KDE']
    DPI = PLOT_SETUP['DPI']
    EXT = PLOT_SETUP['EXTENSION']
    AUX = DATASET['DATASET']
    COLUMN = DATASET['COLUMN']
    DATA = AUX[COLUMN]
    # Plot
    [W, H] = CONVERT_SI_TO_INCHES(W, H)
    sns.set(style = 'ticks')
    FIG, (AX_BOX, AX_HIST) = plt.subplots(2, figsize = (W, H), sharex = True, gridspec_kw = {'height_ratios': (.15, .85)})
    sns.boxplot(DATA, ax = AX_BOX, color = CHART_COLOR)
    sns.histplot(DATA, ax = AX_HIST, kde = KDE, color = CHART_COLOR, bins = BINS)
    AX_BOX.set(yticks = [])
    AX_BOX.set(xlabel='')
    font = {'fontname': 'Arial',
            'color':  LABELS_COLOR,
            'weight': 'normal',
            'size': LABELS_SIZE}
    AX_HIST.set_xlabel(X_AXIS_LABEL, fontdict = font)
    AX_HIST.set_ylabel('Frequência', fontdict = font)
    AX_HIST.tick_params(axis = 'x', labelsize = X_AXIS_SIZE, colors = AXISES_COLOR)
    AX_HIST.tick_params(axis = 'y', labelsize = Y_AXIS_SIZE, colors = AXISES_COLOR)
    sns.despine(ax = AX_HIST)
    sns.despine(ax = AX_BOX, left = True)
    # Save figure
    SAVE_GRAPHIC(NAME, EXT, DPI)

# PLOTAGEM 2
def RASD_PLOT_2(DATASET, PLOT_SETUP):
    """

    This function shows a scatter chart with results between limits and resistances demands

    Input: 
    DATASET     | Results from a RASD Toolboox                             | Py dataframe or Py Numpy array[N_POP x 1]
                |    Dictionary tags                                       |
                |    'DATA'          == Complete data                      | Py Numpy array[N_POP x 1]
                |    'X DATA'        == Dataframe column plots in X axis   | String
                |    'Y DATA'        == Dataframe column plots in Y axis   | String
                |    'HUE VALUE'     == Data plotted in Y                  | String
    PLOT_SETUP  | Contains specifications of each model of chart           | Py dictionary
                |    Dictionary tags                                       |
                |    'NAME'          == Filename output file               | String 
                |    'WIDTH'         == Width figure                       | Float
                |    'HEIGHT         == Height figure                      | Float
                |    'X AXIS SIZE'   == X axis size                        | Float
                |    'Y AXIS SIZE'   == Y axis size                        | Float
                |    'AXISES COLOR'  == Axis color                         | String
                |    'X AXIS LABEL'  == X label name                       | String
                |    'Y AXIS LABEL'  == Y label name                       | String             
                |    'LABELS SIZE'   == Labels size                        | Float
                |    'LABELS COLOR'  == labels color                       | Float
                |    'LOC LEGEND'    == Legend position                    | String
                |    'TITLE LEGEND'  == Text in legend                     | String
                |    'DPI'           == Dots Per Inch - Image quality      | Integer   
                |    'EXTENSION'     == Extension output file              | String ('.svg, '.png', '.eps' or '.pdf')
    
    Output:
    N/A
    """
    # Setup
    NAME = PLOT_SETUP['NAME']
    EXT = PLOT_SETUP['EXTENSION']
    DPI = PLOT_SETUP['DPI']
    W = PLOT_SETUP['WIDTH']
    H = PLOT_SETUP['HEIGHT']
    X_AXIS_SIZE = PLOT_SETUP['X AXIS SIZE']
    Y_AXIS_SIZE = PLOT_SETUP['Y AXIS SIZE']
    AXISES_COLOR = PLOT_SETUP['AXISES COLOR']
    X_AXIS_LABEL = PLOT_SETUP['X AXIS LABEL']
    Y_AXIS_LABEL = PLOT_SETUP['Y AXIS LABEL']
    LABELS_SIZE = PLOT_SETUP['LABELS SIZE']
    LABELS_COLOR = PLOT_SETUP['LABELS COLOR']
    LOC_LEGEND = PLOT_SETUP['LOC LEGEND']
    TITLE_LEGEND = PLOT_SETUP['TITLE LEGEND']
    DATA = DATASET['DATASET']
    X_DATA = DATASET['X DATA']
    Y_DATA = DATASET['Y DATA']
    HUE_VALUE = DATASET['HUE VALUE']
    # Plot
    sns.set(style = 'ticks')
    [W, H] = CONVERT_SI_TO_INCHES(W, H)
    FIG, AX = plt.subplots(figsize = (W, H))
    sns.scatterplot(data = DATA, x = X_DATA, y = Y_DATA, hue = HUE_VALUE,  palette=['orange'])
    font = {'fontname': 'Arial',
        'color':  LABELS_COLOR,
        'weight': 'bold',
        'size': LABELS_SIZE}
    AX.set_xlabel(X_AXIS_LABEL, fontdict = font)
    AX.set_ylabel(Y_AXIS_LABEL, fontdict = font)
    AX.tick_params(axis = 'x', labelsize = X_AXIS_SIZE, colors = AXISES_COLOR)
    AX.tick_params(axis = 'y', labelsize = Y_AXIS_SIZE, colors = AXISES_COLOR)
    AX.legend(loc = LOC_LEGEND, title = TITLE_LEGEND)
    # Save figure
    SAVE_GRAPHIC(NAME, EXT, DPI)

# PLOTAGEM 3
def RASD_PLOT_3(DATASET, PLOT_SETUP):
    """
    This functions plots a scatter chart with variables X and Z in function of G value
    
    Input: 
    DATASET     | Results from a RASD Toolboox                             | Py dataframe or Py Numpy array[N_POP x 1]
                |    Dictionary tags                                       |
                |    'DATA'          == Complete data                      | Py Numpy array[N_POP x 1]
                |    'X DATA'        == Dataframe name column plots in X   | String
                |    'Y DATA'        == Dataframe column plots in Y        | String
                |    'G VALUE'       == Dataframe column plots in C        | String
    PLOT_SETUP  | Contains specifications of each model of chart           | Py dictionary
                |    Dictionary tags                                       |
                |    'NAME'          == Filename output file               | String 
                |    'WIDTH'         == Width figure                       | Float
                |    'HEIGHT         == Height figure                      | Float
                |    'X AXIS SIZE'   == X axis size                        | Float
                |    'Y AXIS SIZE'   == Y axis size                        | Float
                |    'AXISES COLOR'  == Axis color                         | String
                |    'X AXIS LABEL'  == X label name                       | String
                |    'Y AXIS LABEL'  == Y label name                       | String             
                |    'LABELS SIZE'   == Labels size                        | Float
                |    'LABELS COLOR'  == Labels color                       | Float
                |    'TRANSPARENCY'  == Blending value                     | Float
                |    'COLOR MAP'     == Colormap instance, Registered name | String
                |    'DPI'           == Dots Per Inch - Image quality      | Integer   
                |    'EXTENSION'     == Extension output file              | String ('.svg, '.png', '.eps' or '.pdf')
    
    Output:

    EXAMPLE:
    # PLOT SETUP
    PLOT_SETUP = {'NAME': 'WANDER',
                    'EXTENSION': '.svg',
                    'DPI': 600,
                    'WIDTH': 0.20, 
                    'HEIGHT': 0.10,              
                    'X DATA': 'S_0',
                    'Y DATA': 'R_0',
                    'X AXIS SIZE': 20,
                    'Y AXIS SIZE': 20,
                    'AXISES COLOR': '#000000',
                    'X AXIS LABEL': '$S_0$',
                    'Y AXIS LABEL': '$R_0$',
                    'LABELS SIZE': 16,
                    'LABELS COLOR': '#000000',
                    'C VALUE: 'G_0',
                    'TRANSPARENCY': 0.8,
                    'COLOR MAP': 'viridis'}
    # RESULTS
    DATASET = RESULTS_RASD
    # CALL PLOT
    RASD.RASD_PLOT_3(DATASET, PLOT_SETUP)
    """
    # SETUP CHART

    NAME = PLOT_SETUP['NAME']
    EXT = PLOT_SETUP['EXTENSION']
    DPI = PLOT_SETUP['DPI']
    W = PLOT_SETUP['WIDTH']
    H = PLOT_SETUP['HEIGHT']
    X_DATA = DATASET['X DATA']
    Y_DATA = DATASET['Y DATA']
    X_AXIS_SIZE = PLOT_SETUP['X AXIS SIZE']
    Y_AXIS_SIZE = PLOT_SETUP['Y AXIS SIZE']
    AXISES_COLOR = PLOT_SETUP['AXISES COLOR']
    X_AXIS_LABEL = PLOT_SETUP['X AXIS LABEL']
    Y_AXIS_LABEL = PLOT_SETUP['Y AXIS LABEL']
    LABELS_SIZE = PLOT_SETUP['LABELS SIZE']
    LABELS_COLOR = PLOT_SETUP['LABELS COLOR']
    C_VALUE = DATASET['G VALUE']
    TRANSPARENCY = PLOT_SETUP['TRANSPARENCY']
    COLOR_MAP = PLOT_SETUP['COLOR MAP']
    A_UX = DATASET['DATASET']
    # CONVERT UNITS OF SIZE FIGURE
    [W, H] = CONVERT_SI_TO_INCHES(W, H)

    # PLOT

    AUX = plt.Normalize(A_UX[C_VALUE].min(), A_UX[C_VALUE].max())
    FIG, AX = plt.subplots(figsize = (W, H))
    plt.scatter(x = A_UX[X_DATA], y = A_UX[Y_DATA], c = -A_UX[C_VALUE], cmap = COLOR_MAP, alpha = TRANSPARENCY)
    font = {'fontname': 'Arial',
        'color':  LABELS_COLOR,
        'weight': 'bold',
        'size': LABELS_SIZE}
    AX.set_xlabel(X_AXIS_LABEL, fontdict = font)
    AX.set_ylabel(Y_AXIS_LABEL, fontdict = font)
    AX.tick_params(axis = 'x', labelsize = X_AXIS_SIZE, colors = AXISES_COLOR)
    AX.tick_params(axis = 'y', labelsize = Y_AXIS_SIZE, colors = AXISES_COLOR)
    AUX1 =  ScalarMappable(norm = AUX, cmap = COLOR_MAP)
    FIG.colorbar(AUX1, ax = AX)
    # SAVEFIG
    SAVE_GRAPHIC(NAME, EXT, DPI)

# PLOTAGEM 4
def RASD_PLOT_4(DATASET, PLOT_SETUP):
    """
    This function plots two histograms in a single one chart

    Input: 
    DATASET     | Results from a RASD Toolboox                             | Py dataframe or Py Numpy array[N_POP x 1]
                |    Dictionary tags                                       |
                |    'DATA'          == Complete data                      | Py Numpy array[N_POP x 1]
                |    'X DATA'        == Dataframe name column plots in X   | String
                |    'Y DATA'        == Dataframe column plots in Y        | String
                |    'C VALUE'       == Dataframe column plots in C        | String
    PLOT_SETUP  | Contains specifications of each model of chart           | Py dictionary
                |    Dictionary tags                                       |
                |    'NAME'          == Filename output file               | String 
                |    'WIDTH'         == Width figure                       | Float
                |    'HEIGHT         == Height figure                      | Float
                |    'X AXIS SIZE'   == X axis size                        | Float
                |    'Y AXIS SIZE'   == Y axis size                        | Float
                |    'AXISES COLOR'  == Axis color                         | String
                |    'X AXIS LABEL'  == X label name                       | String
                |    'Y AXIS LABEL'  == Y label name                       | String             
                |    'LABELS SIZE'   == Labels size                        | Float
                |    'LABELS COLOR'  == Labels color                       | Float
                |    'ALPHA'         == Blending value                     | Float
                |    'BINS'          == Equal width bins in the range      | Integer
                |    'DPI'           == Dots Per Inch - Image quality      | Integer   
                |    'EXTENSION'     == Extension output file              | String ('.svg, '.png', '.eps' or '.pdf')
    
    Output:

    EXAMPLE:
    # PLOT SETUP
    PLOT_SETUP = {'NAME': 'WANDER',
                    'EXTENSION': '.svg',
                    'DPI': 600,
                    'WIDTH': 0.20,
                    'HEIGHT': 0.10,
                    'X DATA': 'S_0',
                    'Y DATA': 'R_0',
                    'X AXIS SIZE': 20,
                    'Y AXIS SIZE': 20,
                    'AXISES COLOR': '#000000',
                    'X AXIS LABEL': '$S_0 R_0$',
                    'Y AXIS LABEL': '$SOMATÓRIO$',
                    'LABELS SIZE': 16,
                    'LABELS COLOR': '#000000',
                    'C VALUE': 'G_0',
                    'TRANSPARENCY': 0.8,
                    'COLOR MAP': 'viridis',
                    'BINS': '50',
                    'ALPHA': '0.6'}
    # RESULTS
    DATASET = RESULTS_RASD
    # CALL PLOT
    RASD.RASD_PLOT_4(DATASET, PLOT_SETUP)
    """
    # SETUP CHART

    NAME = PLOT_SETUP['NAME']
    EXT = PLOT_SETUP['EXTENSION']
    DPI = PLOT_SETUP['DPI']
    W = PLOT_SETUP['WIDTH']
    H = PLOT_SETUP['HEIGHT']
    X_DATA = DATASET['X DATA']
    Y_DATA = DATASET['Y DATA']
    X_AXIS_SIZE = PLOT_SETUP['X AXIS SIZE']
    Y_AXIS_SIZE = PLOT_SETUP['Y AXIS SIZE']
    AXISES_COLOR = PLOT_SETUP['AXISES COLOR']
    X_AXIS_LABEL = PLOT_SETUP['X AXIS LABEL']
    Y_AXIS_LABEL = PLOT_SETUP['Y AXIS LABEL']
    LABELS_SIZE = PLOT_SETUP['LABELS SIZE']
    LABELS_COLOR = PLOT_SETUP['LABELS COLOR']
    X_DATA_LABEL = PLOT_SETUP['X DATA']
    Y_DATA_LABEL = PLOT_SETUP['Y DATA']
    #C_VALUE = DATASET['C VALUE']
    TRANSPARENCY = PLOT_SETUP['TRANSPARENCY']
    COLOR_MAP = PLOT_SETUP['COLOR MAP']
    BINS = int(PLOT_SETUP['BINS'])
    ALPHA = float(PLOT_SETUP['ALPHA'])
    A_UX = DATASET['DATASET']
    # CONVERT UNITS OF SIZE FIGURE
    [W, H] = CONVERT_SI_TO_INCHES(W, H)

    # PLOT

    plt.subplots(figsize=(W, H))
    plt.hist(A_UX[X_DATA], bins=BINS, label=X_DATA_LABEL, alpha=ALPHA)
    plt.hist(A_UX[Y_DATA], bins=BINS, label=Y_DATA_LABEL, alpha=ALPHA)
    plt.legend()
    plt.xlabel(X_AXIS_LABEL)
    plt.ylabel(Y_AXIS_LABEL)
    #plt.tick_params(axis='x', labelsize=X_AXIS_SIZE, colors=AXISES_COLOR)
    #plt.tick_params(axis='y', labelsize=Y_AXIS_SIZE, colors=AXISES_COLOR)
    #AUX_1 = ScalarMappable(norm=AUX, cmap=COLOR_MAP)
    #FIG.colorbar(AUX_1)
    # SAVEFIG
    SAVE_GRAPHIC(NAME, EXT, DPI)

    # PLOTAGEM 5
def RASD_PLOT_5(DATASET, PLOT_SETUP):
    """
    This function plots a chart with values of number of simulations X Beta/Failure Probability

    Input: 
    DATASET     | Results from a RASD Toolboox                             | Py dataframe or Py Numpy array[N_POP x 1]
                |    Dictionary tags                                       |
                |    'DATA'          == Complete data                      | Py Numpy array[N_POP x 1]
    PLOT_SETUP  | Contains specifications of each model of chart           | Py dictionary
                |    Dictionary tags                                       |
                |    'NAME'          == Filename output file               | String 
                |    'WIDTH'         == Width figure                       | Float
                |    'HEIGHT         == Height figure                      | Float
                |    'X AXIS SIZE'   == X axis size                        | Float
                |    'Y AXIS SIZE'   == Y axis size                        | Float
                |    'AXISES COLOR'  == Axis color                         | String
                |    'X AXIS LABEL'  == X label name                       | String
                |    'Y AXIS LABEL'  == Y label name                       | String             
                |    'LABELS SIZE'   == Labels size                        | Float
                |    'LABELS COLOR'  == Labels color                       | Float
                |    'DPI'           == Dots Per Inch - Image quality      | Integer 
                |    'POPULATION'    == Population array                   | Py Numpy array[N_POP x N_SIMULACOCES]
                |    'TYPE'          == Chart type                         | String ('Beta', 'Pf')
                |    'EXTENSION'     == Extension output file              | String ('.svg, '.png', '.eps' or '.pdf')
    
    Output:

    EXAMPLE:
    # PLOT SETUP
    PLOT_SETUP = {'NAME': 'WANDER',
                    'EXTENSION': '.svg',
                    'DPI': 600,
                    'WIDTH': 0.20,
                    'HEIGHT': 0.10,
                    'X AXIS SIZE': 20,
                    'Y AXIS SIZE': 20,
                    'AXISES COLOR': '#000000',
                    'X AXIS LABEL': 'Número de Simulações (ns)',
                    'Y AXIS LABEL': 'Beta',
                    'LABELS SIZE': 16,
                    'LABELS COLOR': '#000000',
                    'CHART COLOR': 'blue',
                    'POPULATION' : POP,
                    'TYPE' : 'Beta'}
    # RESULTS
    OPCOES_DADOS = {'DATASET': RESULTS_TEST}       

    # CALL PLOT
    RASD_PLOT_5(OPCOES_DADOS, OPCOES_GRAFICAS)
    """

    # SETUP CHART

    NAME = PLOT_SETUP['NAME']
    EXT = PLOT_SETUP['EXTENSION']
    DPI = PLOT_SETUP['DPI']
    W = PLOT_SETUP['WIDTH']
    H = PLOT_SETUP['HEIGHT']
    DATAS = DATASET['DATASET']
    X_AXIS_SIZE = PLOT_SETUP['X AXIS SIZE']
    Y_AXIS_SIZE = PLOT_SETUP['Y AXIS SIZE']
    AXISES_COLOR = PLOT_SETUP['AXISES COLOR']
    X_AXIS_LABEL = PLOT_SETUP['X AXIS LABEL']
    Y_AXIS_LABEL = PLOT_SETUP['Y AXIS LABEL']
    LABELS_SIZE = PLOT_SETUP['LABELS SIZE']
    LABELS_COLOR = PLOT_SETUP['LABELS COLOR']
    CHART_COLOR = PLOT_SETUP['CHART COLOR']
    PF_AUX = []
    BETA_AUX = []
    POP_SIZE = len(PLOT_SETUP['POPULATION'])
    POPULATION = PLOT_SETUP['POPULATION']
    CHART_TYPE = PLOT_SETUP['TYPE']
    # CONVERT UNITS OF SIZE FIGURE
    [W, H] = CONVERT_SI_TO_INCHES(W, H)

    # PLOT

    for i in range(POP_SIZE):
        PF_AUX.append(DATAS[i]['PROBABILITY OF FAILURE'][0])
        BETA_AUX.append(DATAS[i]['BETA INDEX'][0])
    plt.subplots(figsize=(W, H))    
    if CHART_TYPE.upper() == 'PF':
        plt.plot(POPULATION, PF_AUX, color=CHART_COLOR)
    elif CHART_TYPE.upper() == 'BETA':
        plt.plot(POPULATION, BETA_AUX, color=CHART_COLOR)
    else:
        print("Chart type unknow.")
    plt.xlabel(X_AXIS_LABEL)
    plt.ylabel(Y_AXIS_LABEL)
    #plt.tick_params(axis='x', labelsize=X_AXIS_SIZE, colors=AXISES_COLOR)
    #plt.tick_params(axis='y', labelsize=Y_AXIS_SIZE, colors=AXISES_COLOR)
    #AUX_1 = ScalarMappable(norm=AUX, cmap=COLOR_MAP)
    #FIG.colorbar(AUX_1)
    # SAVEFIG
    SAVE_GRAPHIC(NAME, EXT, DPI)

#   /$$$$$$  /$$$$$$$  /$$$$$$$$ /$$$$$$$$       /$$$$$$$$ /$$$$$$$$  /$$$$$$  /$$   /$$ /$$   /$$  /$$$$$$  /$$        /$$$$$$   /$$$$$$  /$$$$$$ /$$$$$$$$  /$$$$$$ 
#  /$$__  $$| $$__  $$| $$_____/| $$_____/      |__  $$__/| $$_____/ /$$__  $$| $$  | $$| $$$ | $$ /$$__  $$| $$       /$$__  $$ /$$__  $$|_  $$_/| $$_____/ /$$__  $$
# | $$  \__/| $$  \ $$| $$      | $$               | $$   | $$      | $$  \__/| $$  | $$| $$$$| $$| $$  \ $$| $$      | $$  \ $$| $$  \__/  | $$  | $$      | $$  \__/
# | $$ /$$$$| $$$$$$$/| $$$$$   | $$$$$            | $$   | $$$$$   | $$      | $$$$$$$$| $$ $$ $$| $$  | $$| $$      | $$  | $$| $$ /$$$$  | $$  | $$$$$   |  $$$$$$ 
# | $$|_  $$| $$____/ | $$__/   | $$__/            | $$   | $$__/   | $$      | $$__  $$| $$  $$$$| $$  | $$| $$      | $$  | $$| $$|_  $$  | $$  | $$__/    \____  $$
# | $$  \ $$| $$      | $$      | $$               | $$   | $$      | $$    $$| $$  | $$| $$\  $$$| $$  | $$| $$      | $$  | $$| $$  \ $$  | $$  | $$       /$$  \ $$
# |  $$$$$$/| $$      | $$$$$$$$| $$$$$$$$         | $$   | $$$$$$$$|  $$$$$$/| $$  | $$| $$ \  $$|  $$$$$$/| $$$$$$$$|  $$$$$$/|  $$$$$$/ /$$$$$$| $$$$$$$$|  $$$$$$/
#  \______/ |__/      |________/|________/         |__/   |________/ \______/ |__/  |__/|__/  \__/ \______/ |________/ \______/  \______/ |______/|________/ \______/ 