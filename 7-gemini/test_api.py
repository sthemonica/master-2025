import google.generativeai as genai
import os
from dotenv import load_dotenv

print("Iniciando teste de conexão com a API...")

try:
    # Carrega a API Key do arquivo .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("API Key não encontrada no arquivo .env")

    print("Configurando a API Key...")
    genai.configure(api_key=api_key)

    print("\nBuscando modelos disponíveis para 'generateContent'...")
    model_list = []
    for model in genai.list_models():
      if 'generateContent' in model.supported_generation_methods:
        model_list.append(model.name)

    if model_list:
        print("\n--- Modelos Disponíveis ---")
        for model_name in sorted(model_list):
            print(model_name)
        print("---------------------------\n")
    else:
        print("Nenhum modelo compatível foi encontrado.")

    # Verificação específica para o gemini-pro-vision
    if 'models/gemini-pro-vision' in model_list:
        print("✅ Boa notícia: 'gemini-pro-vision' está na sua lista de modelos disponíveis!")
    else:
        print("❌ Atenção: 'gemini-pro-vision' NÃO foi encontrado na sua lista de modelos disponíveis.")


except Exception as e:
    print(f"\nOcorreu um erro durante o teste: {e}")