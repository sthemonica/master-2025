import cv2
import numpy as np
import os
from pathlib import Path

# ========================
# CONFIGURAÇÕES
# ========================
INPUT_DIR  = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\3-Datasets\img-lcm\IC-mask-amanda\masked\val"
OUTPUT_DIR = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\3-Datasets\img-lcm\IC-mask-amanda\masked\val_masks"
MIN_AREA   = 50  # ignora manchas muito pequenas

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# FUNÇÃO PARA CRIAR MÁSCARA PREENCHIDA
# ========================
def create_filled_mask(img_path, output_dir):
    # Carregar imagem anotada
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[ERRO] Não consegui ler {img_path}")
        return
    
    # Converter para RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Definir intervalo para vermelho (ajuste se necessário)
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 80, 80])

    # Criar máscara para vermelho
    mask = cv2.inRange(img_rgb, lower_red, upper_red)

    # Fechar pequenos buracos da borda (ajuda a evitar vazamentos)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar máscara preenchida
    filled_mask = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= MIN_AREA:  # ignora manchas muito pequenas
            cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Salvar no diretório de saída
    stem = Path(img_path).stem
    save_path = Path(output_dir) / f"{stem}_mask.png"
    cv2.imwrite(str(save_path), filled_mask)

    print(f"[OK] Máscara criada: {save_path}")

# ========================
# LOOP EM TODAS AS IMAGENS
# ========================
def main():
    input_path = Path(INPUT_DIR)
    files = []
    for ext in ("*.bmp", "*.BMP"):
        files.extend(input_path.rglob(ext))  # busca recursiva

    print(f"[INFO] Encontradas {len(files)} imagens.")
    for img_path in files:
        create_filled_mask(img_path, OUTPUT_DIR)

    print(f"[FIM] Processadas: {len(files)} imagens. Saída: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
