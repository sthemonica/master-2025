import pandas as pd
import numpy as np
import cv2
import os
import re
import ast

def parse_coordenadas(coord_str):
    """
    Função para extrair uma ou mais listas de polígonos da string de coordenadas.
    Usa expressões regulares para encontrar todos os polígonos.
    """
    # A expressão regular encontra todos os padrões "poligono_xy": [[...]]
    # e captura apenas o conteúdo dentro dos colchetes externos.
    matches = re.findall(r'"poligono_xy":\s*(\[\[.*?\]\])', coord_str)
    
    poligonos = []
    for match in matches:
        try:
            # ast.literal_eval é uma forma segura de converter uma string de lista/dicionário Python
            # para o objeto Python correspondente.
            poligono = ast.literal_eval(match)
            poligonos.append(poligono)
        except (ValueError, SyntaxError):
            # Ignora polígonos malformados se houver algum
            print(f"Aviso: Não foi possível processar o polígono: {match}")
            continue
            
    return poligonos

# --- CONFIGURAÇÃO ---
# Altere estes caminhos conforme a sua estrutura de arquivos
CSV_PATH = 'resultados_segmentacao_v2.csv' # Caminho para o seu arquivo CSV
OUTPUT_MASKS_DIR = 'mascaras_geradas'      # Pasta onde as máscaras serão salvas

# --- LÓGICA PRINCIPAL ---

# 1. Criar o diretório de saída se ele não existir
if not os.path.exists(OUTPUT_MASKS_DIR):
    os.makedirs(OUTPUT_MASKS_DIR)
    print(f"Diretório '{OUTPUT_MASKS_DIR}' criado.")

# 2. Carregar o CSV com o pandas
try:
    df = pd.read_csv(CSV_PATH)
    print(f"CSV '{CSV_PATH}' carregado com sucesso. Total de {len(df)} linhas.")
except FileNotFoundError:
    print(f"Erro: O arquivo CSV '{CSV_PATH}' não foi encontrado. Verifique o caminho.")
    exit()

# 3. Iterar sobre cada linha do DataFrame para processar cada imagem
for index, row in df.iterrows():
    caminho_imagem = row['caminho_da_imagem']
    nome_imagem = row['nome_da_imagem']
    coordenadas_str = row['coordenadas_poligonos']
    
    # Verifica se o caminho da imagem original existe
    if not os.path.exists(caminho_imagem):
        print(f"Aviso: Imagem não encontrada em '{caminho_imagem}'. Pulando linha {index + 2}.")
        continue

    try:
        # 4. Ler a imagem original para obter suas dimensões (altura e largura)
        img_original = cv2.imread(caminho_imagem)
        if img_original is None:
            print(f"Aviso: Falha ao ler a imagem '{caminho_imagem}'. Pulando.")
            continue
            
        altura, largura, _ = img_original.shape

        # 5. Criar uma máscara preta (array de zeros) com as mesmas dimensões da imagem original
        # A máscara será de canal único (escala de cinza)
        mascara = np.zeros((altura, largura), dtype=np.uint8)

        # 6. Extrair os polígonos da string
        poligonos = parse_coordenadas(coordenadas_str)

        if not poligonos:
            print(f"Aviso: Nenhum polígono válido encontrado para a imagem '{nome_imagem}'. Máscara ficará preta.")
        else:
            # Converte os pontos dos polígonos para o formato que o OpenCV espera
            # (uma lista de arrays NumPy de inteiros de 32 bits)
            poligonos_np = [np.array(p, dtype=np.int32) for p in poligonos]
            
            # 7. Desenhar os polígonos preenchidos na máscara
            # A cor 255 representa o branco, que é o padrão para máscaras binárias
            cv2.fillPoly(mascara, poligonos_np, color=255)

        # 8. Salvar a máscara gerada
        # É uma boa prática salvar máscaras em formato sem perdas, como .png
        nome_mascara = os.path.splitext(nome_imagem)[0] + '.png'
        caminho_saida = os.path.join(OUTPUT_MASKS_DIR, nome_mascara)
        cv2.imwrite(caminho_saida, mascara)
        
        print(f"Máscara para '{nome_imagem}' salva em '{caminho_saida}'.")

    except Exception as e:
        print(f"Ocorreu um erro ao processar a imagem '{nome_imagem}': {e}")

print("\nProcesso concluído!")