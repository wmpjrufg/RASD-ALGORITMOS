import pandas as pd
import math
import numpy as np


def CALCULO_KV(SOLO):
  # Calculo deformabilidade do solo Es

  # Es = alfa * k * nspt 

  # Entrada
  soil_type = SOLO["TIPO_SOLO"]    
  nspt =  SOLO["NSPT"]   
  carga_pilar = SOLO["CARGA_PILAR"] 
  base_pilar = SOLO["BASE_PILAR"] 
  largura_pilar = SOLO["LARGURA_PILAR"]

  # Solo, a, k(kPa), n
  soil_types = [('Areia', '3', '900', '0.3'), ('Areia Argilo Siltosa', '3', '600', '0.3'),
                ('Areia Argilosa', '3', '550', '0.3'), ('Areia Silto Argilosa', '3', '575', '0.3'),
                ('Areia Siltosa', '3', '700', '0.3'), ('Argila', '7', '200', '0.45'),
                ('Argila Areno Siltosa', '6', '250', '0.45'), ('Argila Arenosa', '6', '300', '0.45'),
                ('Argila Silto Arenosa', '6', '250', '0.45'), ('Argila Siltosa', '6', '200', '0.45'),
                ('Silte', '5', '350', '0.4'), ('Silte Areno Argiloso', '5', '450', '0.4'),
                ('Silte Arenoso', '5', '450', '0.4'), ('Silte Argiloso Arenoso', '5', '325', '0.4'),
                ('Silte Argiloso', '5', '250', '0.4')]

  soil_types_df = pd.DataFrame(soil_types)
  soil_types_df.columns = ['Solo', 'alfa', 'k(kPa)', 'n']
  aux_solo = soil_types_df[(soil_types_df['Solo']==soil_type)]
  soil_alfa = aux_solo['alfa']
  soil_k = aux_solo['k(kPa)']
  soil_n = aux_solo['n']
  Es = float(soil_alfa) * float(soil_k) * float(nspt)

  #print(Es)

  # Tensão admissivel do solo https://suporte.altoqi.com.br/hc/pt-br/articles/360004276094-Como-obter-a-press%C3%A3o-admiss%C3%ADvel-a-partir-do-SPT
  nsptt = float(nspt) / 3 # 3 devido a profundidade do bulbo de tensao analisado
  soil_tension = nsptt /  5 #kgf/cm2

  # Area sapata  https://wwwp.feb.unesp.br/pbastos/concreto3/Sapatas.pdf
  # soil_tension = 0.026 valor teste
  sloped_footing_area = 1.1 * carga_pilar / (soil_tension * 1000) # vezes mil para colocar kN, resultado em cm2

  # Dimensoes da sapata retangular L>=B

  sloped_footing_base = 1/2*( base_pilar - largura_pilar)+(1/4 * ((base_pilar - largura_pilar) ** 2) + sloped_footing_area * 1000) ** (1/2) #menor lado
  sloped_footing_base  = (math.ceil(sloped_footing_base / 5) * 5)
  sloped_footing_length = (largura_pilar - base_pilar + sloped_footing_base)

  new_footing_area = (sloped_footing_length * sloped_footing_base) / 1000


  if new_footing_area > sloped_footing_area:
    sloped_footing_area = new_footing_area

  # Constante elastica da mola final 
  kv = 1.33 * Es / ((((sloped_footing_base / 100) ** 2) * (sloped_footing_length / 100)) ** (1/3))
  kv_final_k = kv * sloped_footing_area / 1000 # kN/m2 dividido por 10000 para tirar de cm para m  
  kv_final = kv_final_k * 1000 # N/m2 vezes 1000 para tirar do kN

  return(kv_final)