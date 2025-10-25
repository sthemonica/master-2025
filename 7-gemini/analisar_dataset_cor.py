import google.generativeai as genai
import os
import pathlib
import csv
import json
from dotenv import load_dotenv
import PIL.Image
from PIL import ImageEnhance # Importa a ferramenta para melhorar a imagem

# --- CONFIGURAÇÃO INICIAL ---

# Carrega a API Key do arquivo .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Erro: A chave de API do Gemini não foi encontrada no arquivo .env.")

genai.configure(api_key=api_key)

# Usando o modelo que já confirmamos estar disponível
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Define as pastas que contêm as imagens
pastas_imagens = ['train', 'val']

# Nome do arquivo CSV de saída
arquivo_saida_csv = 'resultados_segmentacao_v2.csv' # Salvar em um novo arquivo para comparar

# --- MELHORIA 1: ENGENHARIA DE PROMPT ---
prompt_para_ia = """
Aja como um especialista em biologia celular e análise de imagens de microscopia.
Sua tarefa é realizar a segmentação de instâncias na imagem de fotomicrografia fornecida.

Analise a imagem cuidadosamente. As células podem ter baixo contraste, podem estar agrupadas, sobrepostas ou ter formatos variados. O fundo pode conter ruído ou detritos. Sua meta é identificar e delinear o contorno de cada objeto que seja uma célula provável.

Pense passo a passo: primeiro, localize as regiões de interesse que contêm células. Em segundo lugar, para cada célula, trace o polígono que define sua borda. Por fim, formate a saída final.

Sua resposta deve ser **EXCLUSIVAMENTE** um objeto JSON válido. Não inclua nenhum texto, explicação ou formatação fora do bloco JSON.

A estrutura do JSON deve ser:
{
  "descricao": "Uma breve descrição técnica da imagem, como o tipo de células ou a qualidade da imagem.",
  "celulas": [
    {
      "poligono_xy": [[x1, y1], [x2, y2], [x3, y3], ...]
    }
  ]
}

Se, mesmo após uma análise cuidadosa, nenhuma célula for detectada, retorne uma lista vazia para "celulas".
"""

# --- MELHORIA 2: AJUSTE DE PARÂMETROS DO MODELO ---
# Temperatura baixa (ex: 0.1) torna o modelo mais determinístico e focado.
generation_config = genai.types.GenerationConfig(
    temperature=0.1 
)

# --- LÓGICA PRINCIPAL ---

def processar_imagens():
    print("Iniciando o processo de análise de imagens (Versão Melhorada)...")

    lista_de_imagens = []
    extensoes_suportadas = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for pasta in pastas_imagens:
        for extensao in extensoes_suportadas:
            lista_de_imagens.extend(list(pathlib.Path(pasta).glob(extensao)))

    if not lista_de_imagens:
        print("Nenhuma imagem encontrada.")
        return

    total_imagens = len(lista_de_imagens)
    print(f"Total de {total_imagens} imagens encontradas. Processando...")

    with open(arquivo_saida_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['nome_da_imagem', 'caminho_da_imagem', 'coordenadas_poligonos', 'descricao_imagem'])

        for i, caminho_imagem in enumerate(lista_de_imagens):
            print(f"Processando imagem {i + 1}/{total_imagens}: {caminho_imagem.name}")
            try:
                img_original = PIL.Image.open(caminho_imagem)

                # --- MELHORIA 3: PRÉ-PROCESSAMENTO DE IMAGEM (AUMENTO DE CONTRASTE) ---
                # Aumenta o contraste em 50% para tornar as células mais visíveis
                enhancer = ImageEnhance.Contrast(img_original)
                img_processada = enhancer.enhance(1.5)

                # Envia a imagem processada (com mais contraste) para a API
                response = model.generate_content(
                    [prompt_para_ia, img_processada],
                    generation_config=generation_config # Aplica a configuração de temperatura
                )
                
                resposta_limpa = response.text.strip()
                if resposta_limpa.startswith("```json"):
                    resposta_limpa = resposta_limpa[7:]
                if resposta_limpa.endswith("```"):
                    resposta_limpa = resposta_limpa[:-3]
                
                dados = json.loads(resposta_limpa)
                descricao = dados.get('descricao', 'N/A')
                celulas = dados.get('celulas', [])
                coordenadas_str = json.dumps(celulas)
                writer.writerow([
                    caminho_imagem.name,
                    str(caminho_imagem.resolve()),
                    coordenadas_str,
                    descricao
                ])
            except Exception as e:
                print(f"  -> ERRO ao processar a imagem {caminho_imagem.name}: {e}")
                writer.writerow([caminho_imagem.name, str(caminho_imagem.resolve()), 'ERRO', str(e)])

    print(f"\nProcesso concluído! Resultados salvos em '{arquivo_saida_csv}'.")

# --- EXECUÇÃO ---
if __name__ == '__main__':
    processar_imagens()