# xai_resnet_fold5.py
# Script para aplicar XAI (Grad-CAM, IG, Occlusion, DeepLIFT) na ResNet50 fold 5
# em todas as imagens de val/{infected,uninfected} do dataset LCM.

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2

from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms
from captum.attr import (
    IntegratedGradients,
    Occlusion,
    LayerGradCam,
    LayerAttribution,
    DeepLift,
)

# ================== CONFIGURAÇÕES ==================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_DIR = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\3-Datasets\img-lcm\img_dir_annot_nii\val"
OUTPUT_DIR = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\3-Datasets\img-lcm\xai_resnet_fold5_nii"

# ajuste se o caminho do checkpoint for diferente:
RESNET_CKPT_PATH = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\4-redes\2.resnet\normal\modelos_salvos_resnet\resnet_fold5.pth"


CLASS_NAMES = ["uninfected", "infected"]  # índice 0 -> uninfected, 1 -> infected

# subpastas de saída para cada método XAI
METHOD_DIRS = {
    "gradcam": os.path.join(OUTPUT_DIR, "gradcam"),
    "ig": os.path.join(OUTPUT_DIR, "integrated_gradients"),
    "occlusion": os.path.join(OUTPUT_DIR, "occlusion"),
    #"deeplift": os.path.join(OUTPUT_DIR, "deeplift"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for d in METHOD_DIRS.values():
    os.makedirs(d, exist_ok=True)

# ================== MODELO & TRANSFORMS ==================


def load_resnet_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes

    state = torch.load(RESNET_CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    # 🔧 Desativar inplace em TODOS os ReLUs
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    # 🔍 Verificação: imprime se sobrou algum ReLU inplace=True
    inplace_list = [m.inplace for m in model.modules() if isinstance(m, nn.ReLU)]
    print("Algum ReLU inplace=True ainda?", any(inplace_list))

    model.to(DEVICE)
    model.eval()
    return model




normalize_imagenet = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

transform_gray = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize_imagenet,
    ]
)


def load_image(path):
    img = Image.open(path)
    x = transform_gray(img)
    return img, x.unsqueeze(0).to(DEVICE)


# ================== HELPERS VISUAL ==================


def save_overlay(pil_img, heatmap_tensor, out_path, alpha=0.5):
    """
    pil_img: PIL original
    heatmap_tensor: tensor (1,1,H,W) normalizado [0,1]
    """
    hmap = heatmap_tensor.squeeze().numpy()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    hmap = (hmap * 255).astype(np.uint8)

    img = np.array(pil_img.resize((hmap.shape[1], hmap.shape[0])))

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, hmap_color, alpha, 0)
    cv2.imwrite(out_path, overlay)


# ================== FUNÇÕES XAI ==================


def explain_ig(model, x, target=None, n_steps=50, baseline=None):
    model.zero_grad()
    model.eval()

    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1).item()

    if baseline is None:
        baseline = torch.zeros_like(x)

    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(
        x,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    attr = attributions.abs().sum(dim=1, keepdim=True)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    return attr.detach().cpu()


def explain_occlusion(model, x, target=None, window=15, stride=8, baseline=0):
    model.zero_grad()
    model.eval()

    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1).item()

    occ = Occlusion(model)
    attributions = occ.attribute(
        x,
        strides=(3, stride, stride),
        sliding_window_shapes=(3, window, window),
        baselines=baseline,
        target=target,
    )

    attr = attributions.mean(dim=1, keepdim=True)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    return attr.detach().cpu()


def explain_gradcam(model, gradcam_obj, x, target=None):
    model.zero_grad()
    model.eval()

    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1).item()

    attributions = gradcam_obj.attribute(x, target=target)
    upsampled = LayerAttribution.interpolate(attributions, x.shape[2:])
    attr = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
    return attr.detach().cpu()


# def explain_deeplift(model, x, target=None, baseline=None):
#     model.zero_grad()
#     model.eval()

#     if target is None:
#         with torch.no_grad():
#             target = model(x).argmax(dim=1).item()

#     if baseline is None:
#         baseline = torch.zeros_like(x)

#     dl = DeepLift(model)
#     attributions = dl.attribute(x, baselines=baseline, target=target)

#     attr = attributions.abs().sum(dim=1, keepdim=True)
#     attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
#     return attr.detach().cpu()


# ================== LOOP PRINCIPAL ==================


def collect_samples(val_dir):
    samples = []
    for label_name in ["infected", "uninfected"]:
        class_dir = os.path.join(val_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        label_idx = CLASS_NAMES.index(label_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                fpath = os.path.join(class_dir, fname)
                samples.append((fpath, label_idx))
    random.shuffle(samples)
    return samples


def main():
    model = load_resnet_model()
    print("Modelo ResNet fold5 carregado.")

    # Grad-CAM: usar última camada convolucional (layer4[-1])
    gradcam_layer = model.layer4[-1]
    gradcam_obj = LayerGradCam(model, gradcam_layer)

    samples = collect_samples(VAL_DIR)
    print(f"Total de imagens encontradas: {len(samples)}")

    results = []

    for idx, (img_path, label_idx) in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] Processando: {img_path}")

        pil_img, x = load_image(img_path)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            true_label = CLASS_NAMES[label_idx]
            correct = int(pred_idx == label_idx)

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # XAI: Grad-CAM
        gc_attr = explain_gradcam(model, gradcam_obj, x, target=pred_idx)
        gc_path = os.path.join(METHOD_DIRS["gradcam"], f"{base_name}_gradcam.png")
        save_overlay(pil_img, gc_attr, gc_path)

        # XAI: Integrated Gradients (baseline preto)
        ig_attr = explain_ig(model, x, target=pred_idx, baseline=torch.zeros_like(x))
        ig_path = os.path.join(METHOD_DIRS["ig"], f"{base_name}_ig.png")
        save_overlay(pil_img, ig_attr, ig_path)

        # XAI: Occlusion
        occ_attr = explain_occlusion(model, x, target=pred_idx)
        occ_path = os.path.join(METHOD_DIRS["occlusion"], f"{base_name}_occ.png")
        save_overlay(pil_img, occ_attr, occ_path)

        # XAI: DeepLIFT (baseline preto)
        # try:
        #     dl_attr = explain_deeplift(model, x, target=pred_idx, baseline=torch.zeros_like(x))
        #     dl_path = os.path.join(METHOD_DIRS["deeplift"], f"{base_name}_deeplift.png")
        #     save_overlay(pil_img, dl_attr, dl_path)
        # except RuntimeError as e:
        #     print(f"Erro ao calcular DeepLIFT para {img_path}: {e}")
        #     dl_path = ""

        results.append(
            {
                "image_path": img_path,
                "true_label": true_label,
                "true_idx": label_idx,
                "pred_label": pred_label,
                "pred_idx": pred_idx,
                "correct": correct,
                "prob_uninfected": float(probs[0]),
                "prob_infected": float(probs[1]),
                "gradcam_path": gc_path,
                "ig_path": ig_path,
                "occlusion_path": occ_path,
                #"deeplift_path": dl_path,
            }
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "xai_resnet_fold5_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nArquivo CSV salvo em: {csv_path}")
    print("Processo concluído.")


if __name__ == "__main__":
    main()
