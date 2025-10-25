import google.generativeai as genai
import os
import pathlib
import csv
import json
from dotenv import load_dotenv
import PIL.Image

# --- CONFIGURAÇÃO INICIAL ---

# Carrega a API Key do arquivo .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Erro: A chave de API do Gemini não foi encontrada no arquivo .env.")

genai.configure(api_key=api_key)

# --- ALTERAÇÃO FINAL AQUI ---
# Usando o melhor modelo disponível na sua lista atualizada.
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Define as pastas que contêm as imagens
pastas_imagens = ['train', 'val']

# Nome do arquivo CSV de saída
arquivo_saida_csv = 'resultados_segmentacao.csv'

# --- O PROMPT ESTRUTURADO ---
prompt_para_ia = """
Analise a imagem de células fornecida. Sua tarefa é identificar cada célula visível e fornecer uma descrição geral da imagem.

Sua resposta deve ser **EXCLUSIVAMENTE** um objeto JSON válido, sem nenhum texto ou formatação extra antes ou depois dele.

O objeto JSON deve ter a seguinte estrutura:
{
  "descricao": "Uma breve descrição técnica da imagem, como o tipo de células ou a qualidade da imagem.",
  "celulas": [
    {
      "poligono_xy": [[x1, y1], [x2, y2], [x3, y3], ...]
    }
  ]
}

- A chave "descricao" deve conter a descrição.
- A chave "celulas" deve conter uma lista, onde cada item representa uma célula e seu contorno poligonal em uma lista de coordenadas [x, y].
Se nenhuma célula for detectada, retorne uma lista vazia para "celulas".
"""

# --- LÓGICA PRINCIPAL ---

def processar_imagens():
    print("Iniciando o processo de análise de imagens...")

    lista_de_imagens = []
    extensoes_suportadas = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for pasta in pastas_imagens:
        for extensao in extensoes_suportadas:
            lista_de_imagens.extend(list(pathlib.Path(pasta).glob(extensao)))

    if not lista_de_imagens:
        print("Nenhuma imagem encontrada nas pastas 'train' e 'val'. Verifique sua estrutura de arquivos.")
        return

    total_imagens = len(lista_de_imagens)
    print(f"Total de {total_imagens} imagens encontradas. Processando...")

    with open(arquivo_saida_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['nome_da_imagem', 'caminho_da_imagem', 'coordenadas_poligonos', 'descricao_imagem'])

        # Loop 'for' simples com contador
        for i, caminho_imagem in enumerate(lista_de_imagens):
            # Imprime o progresso no terminal
            print(f"Processando imagem {i + 1}/{total_imagens}: {caminho_imagem.name}")
            try:
                img = PIL.Image.open(caminho_imagem)
                response = model.generate_content([prompt_para_ia, img])
                
                # Limpeza extra para garantir que apenas o JSON seja processado
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