import google.generativeai as genai
import os
from dotenv import load_dotenv
import PIL.Image # Importa a biblioteca de imagem

# Carrega as variáveis de ambiente
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Erro: A chave de API do Gemini não foi encontrada.")
else:
    genai.configure(api_key=api_key)

    # --- MUDANÇA IMPORTANTE AQUI ---
    # Usando o modelo de visão que aceita imagens e texto
    model = genai.GenerativeModel('gemini-pro-vision') 
    
    print("="*50)
    print("🤖 Análise de Imagens com Gemini Vision!")
    print("Deixe em branco e pressione Enter para sair.")
    print("="*50)

    while True:
        # Pede ao usuário o caminho do arquivo de imagem
        caminho_imagem = input("Digite o caminho da imagem (ex: celulas.jpg): ")

        if not caminho_imagem:
            print("\n🤖 Até a próxima!")
            break

        try:
            # Tenta abrir a imagem que o usuário forneceu
            img = PIL.Image.open(caminho_imagem)

            # Pergunta o que fazer com a imagem
            prompt_texto = input("O que você quer que eu faça com esta imagem? ")

            print("\nAnalisando a imagem... aguarde um momento.\n")

            # Envia tanto o texto quanto a imagem para o modelo
            response = model.generate_content([prompt_texto, img])

            # Imprime a resposta
            print("="*20 + " RESPOSTA DO GEMINI " + "="*20)
            print(response.text)
            print("="*62 + "\n")

        except FileNotFoundError:
            print("\nERRO: Arquivo de imagem não encontrado. Verifique o caminho e tente novamente.\n")
        except Exception as e:
            print(f"\nOcorreu um erro inesperado: {e}\n")