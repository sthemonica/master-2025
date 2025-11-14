import io
import numpy as np
import cv2
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.swin_transformer import swin_t

from captum.attr import (
    IntegratedGradients,
    Occlusion,
    LayerGradCam,
    LayerAttribution,
)

from cellpose import models as cp_models, core as cp_core
from skimage import measure
import pandas as pd  # <--- NOVO: para a tabela


# =========================================================
# CONFIG GERAL
# =========================================================
st.set_page_config(page_title="Malaria XAI Pipeline", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ajuste esses caminhos para seus arquivos locais .pth
RESNET_CKPT_PATH = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\4-redes\2.resnet\normal\modelos_salvos_resnet\resnet_fold5.pth"
SWIN_CKPT_PATH = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\4-redes\3.swin\swin-sem-optuna\modelos_salvos_swin\swin_fold2.pth"

CLASS_NAMES = ["uninfected", "infected"]  # índice 0 -> uninfected, 1 -> infected


# =========================================================
# TRANSFORMS (mesmo padrão dos scripts de XAI)
# =========================================================
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

def load_image_pil(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")

def tensor_from_pil_gray(pil_img):
    x = transform_gray(pil_img)
    return x.unsqueeze(0).to(DEVICE)  # [1,3,224,224]


# =========================================================
# MODELOS + GRAD-CAM (ADAPTADO DOS SEUS SCRIPTS)
# =========================================================
def load_resnet_model(ckpt_path: str):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    inplace_list = [m.inplace for m in model.modules() if isinstance(m, nn.ReLU)]
    print("Algum ReLU inplace=True ainda?", any(inplace_list))

    model.to(DEVICE)
    model.eval()

    gradcam_layer = model.layer4[-1]
    gradcam_obj = LayerGradCam(model, gradcam_layer)

    return model, gradcam_obj


def load_swin_model(ckpt_path: str):
    swin = swin_t(weights=None)
    num_features = swin.head.in_features
    swin.head = nn.Linear(num_features, 2)

    state = torch.load(ckpt_path, map_location=DEVICE)
    swin.load_state_dict(state)

    swin.to(DEVICE)
    swin.eval()

    swin_layer = swin.features[0][0]
    gradcam_obj = LayerGradCam(swin, swin_layer)

    return swin, gradcam_obj


def load_custom_model_from_pth(pth_file, base_model: str):
    state = torch.load(io.BytesIO(pth_file.read()), map_location=DEVICE)

    if base_model == "ResNet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(state)

        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        model.to(DEVICE)
        model.eval()
        gradcam_layer = model.layer4[-1]
        gradcam_obj = LayerGradCam(model, gradcam_layer)

    elif base_model == "Swin-T":
        model = swin_t(weights=None)
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 2)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        swin_layer = model.features[0][0]
        gradcam_obj = LayerGradCam(model, swin_layer)

    else:
        raise ValueError("Modelo base não suportado para custom .pth")

    return model, gradcam_obj


# =========================================================
# FUNÇÕES XAI (COPIADAS/ADAPTADAS DOS SEUS SCRIPTS)
# =========================================================
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
    return attr.detach().cpu()  # (1,1,H,W)


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
    return attr.detach().cpu()  # (1,1,H,W)


def explain_gradcam(model, gradcam_obj, x, target=None):
    model.zero_grad()
    model.eval()

    if target is None:
        with torch.no_grad():
            target = model(x).argmax(dim=1).item()

    attributions = gradcam_obj.attribute(x, target=target)
    upsampled = LayerAttribution.interpolate(attributions, x.shape[2:])
    attr = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
    return attr.detach().cpu()  # (1,1,H,W)


def run_xai_heatmap(model, gradcam_obj, x, method="GradCAM", target=None):
    """
    x: tensor [1,3,224,224] (já normalizado)
    return:
      - heatmap (224,224) np.float32 entre 0-1
      - target_class (int)
      - prob_target (float)
    """
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        if target is None:
            target = int(probs.argmax(dim=1).item())
        prob_target = float(probs[0, target].item())

    if method == "GradCAM":
        attr = explain_gradcam(model, gradcam_obj, x, target=target)
    elif method == "Integrated Gradients":
        attr = explain_ig(model, x, target=target, baseline=torch.zeros_like(x))
    elif method == "Occlusion":
        attr = explain_occlusion(model, x, target=target)
    else:
        attr = explain_gradcam(model, gradcam_obj, x, target=target)

    hmap = attr.squeeze().numpy()
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    return hmap.astype(np.float32), target, prob_target


# =========================================================
# CELLPOSE - SEGMENTAÇÃO + CROPS 224x224
# =========================================================
CELLPOSE_MODEL = None

def get_cellpose_model():
    global CELLPOSE_MODEL
    if CELLPOSE_MODEL is None:
        gpu = cp_core.use_gpu()
        CELLPOSE_MODEL = cp_models.CellposeModel(
            gpu=gpu,
            model_type="cyto2"
        )
        print(f"Cellpose inicializado (gpu={gpu})")
    return CELLPOSE_MODEL


def run_cellpose_and_crop(image_pil, min_area=80, max_cells=150):
    model = get_cellpose_model()
    img_rgb = np.array(image_pil)

    channels = [0, 0]

    masks, flows, styles = model.eval(
        img_rgb,
        channels=channels,
        batch_size=8,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize={"tile_norm_blocksize": 0},
    )

    regions = measure.regionprops(masks)
    crops = []

    for reg in regions:
        if reg.area < min_area:
            continue
        y0, x0, y1, x1 = reg.bbox
        crop = image_pil.crop((x0, y0, x1, y1))
        crop_224 = crop.resize((224, 224))

        crops.append({
            "crop": crop_224,
            "bbox": (y0, x0, y1, x1),
            "label": reg.label
        })

        if len(crops) >= max_cells:
            break

    if masks.max() > 0:
        mask_viz = (masks.astype(np.float32) / masks.max() * 255)
    else:
        mask_viz = masks.astype(np.float32)
    mask_viz = mask_viz.astype(np.uint8)
    mask_viz_pil = Image.fromarray(mask_viz)

    return crops, mask_viz_pil


# =========================================================
# RECONSTRUÇÃO HEATMAP FULL
# =========================================================
def reconstruct_full_heatmap(original_pil, cells_with_heatmaps):
    orig = np.array(original_pil).astype(np.float32) / 255.0
    h, w, _ = orig.shape

    heat_canvas = np.zeros((h, w), dtype=np.float32)

    for cell in cells_with_heatmaps:
        y0, x0, y1, x1 = cell["bbox"]
        h_box = y1 - y0
        w_box = x1 - x0
        heat = cell["heatmap"]

        heat_resized = cv2.resize(heat, (w_box, h_box))
        heat_canvas[y0:y1, x0:x1] = np.maximum(
            heat_canvas[y0:y1, x0:x1],
            heat_resized
        )

    if heat_canvas.max() > 0:
        heat_canvas = heat_canvas / heat_canvas.max()

    heat_color = cv2.applyColorMap(
        (heat_canvas * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB) / 255.0

    alpha = 0.5
    overlay = (1 - alpha) * orig + alpha * heat_color
    overlay = np.clip(overlay, 0, 1)

    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
    return overlay_pil


# =========================================================
# STREAMLIT STATE
# =========================================================
if "cells" not in st.session_state:
    st.session_state.cells = None
if "cells_with_heatmaps" not in st.session_state:
    st.session_state.cells_with_heatmaps = None
if "mask_viz" not in st.session_state:
    st.session_state.mask_viz = None

st.title("🦟 Malaria XAI Pipeline - Cellpose + ResNet/Swin + XAI")

st.sidebar.header("Configurações")

uploaded_img = st.sidebar.file_uploader(
    "Envie uma imagem (BMP/JPG/PNG)",
    type=["bmp", "jpg", "jpeg", "png"]
)

# --------------------------
# Exibir imagem original
# --------------------------
if uploaded_img is not None:
    original_img = load_image_pil(uploaded_img)
    st.subheader("Imagem original")
    st.image(original_img, use_container_width=True)

    if st.button("Rodar Cellpose e recortar células (224x224)"):
        with st.spinner("Rodando Cellpose e gerando crops..."):
            cells, mask_viz = run_cellpose_and_crop(original_img)
            st.session_state.cells = cells
            st.session_state.mask_viz = mask_viz
        st.success(f"{len(st.session_state.cells)} células geradas!")

# --------------------------
# Mostrar máscara Cellpose
# --------------------------
if st.session_state.mask_viz is not None:
    st.subheader("Máscara gerada pelo Cellpose (viz simples)")
    st.image(st.session_state.mask_viz, use_container_width=True)

# --------------------------
# Mostrar células
# --------------------------
if st.session_state.cells is not None:
    st.subheader("Células detectadas (crops 224x224)")
    cols = st.columns(4)
    for i, cell in enumerate(st.session_state.cells):
        with cols[i % 4]:
            st.image(cell["crop"], caption=f"Célula {i}", use_container_width=True)

    st.markdown("---")

    # --------------------------
    # Escolha de modelo
    # --------------------------
    st.sidebar.subheader("Modelo de classificação")

    model_option = st.sidebar.selectbox(
        "Escolha um modelo pré-definido",
        ["ResNet50 fold5", "Swin-T fold2"]
    )

    use_custom_model = st.sidebar.checkbox("Quero enviar um modelo .pth")
    custom_model_file = None
    base_model_for_custom = None

    if use_custom_model:
        custom_model_file = st.sidebar.file_uploader(
            "Envie o arquivo .pth do modelo",
            type=["pth"]
        )
        base_model_for_custom = st.sidebar.selectbox(
            "Modelo base do .pth",
            ["ResNet50", "Swin-T"]
        )

    # --------------------------
    # Escolha XAI
    # --------------------------
    xai_method = st.sidebar.selectbox(
        "Método de XAI",
        ["GradCAM", "Integrated Gradients", "Occlusion"]
    )

    # --------------------------
    # Inferência + XAI nas células
    # --------------------------
    if st.button("Rodar inferência + XAI nas células"):
        if use_custom_model and custom_model_file is None:
            st.error("Por favor, envie o arquivo .pth do modelo customizado.")
        else:
            with st.spinner("Carregando modelo..."):
                if use_custom_model:
                    model, gradcam_obj = load_custom_model_from_pth(
                        custom_model_file,
                        base_model=base_model_for_custom
                    )
                else:
                    if model_option == "ResNet50 fold5":
                        model, gradcam_obj = load_resnet_model(RESNET_CKPT_PATH)
                    else:
                        model, gradcam_obj = load_swin_model(SWIN_CKPT_PATH)

            cells_with_heatmaps = []
            for idx, cell in enumerate(st.session_state.cells):
                crop_pil = cell["crop"]
                x = tensor_from_pil_gray(crop_pil)  # [1,3,224,224]

                heatmap, target_class, prob_target = run_xai_heatmap(
                    model, gradcam_obj, x, method=xai_method
                )

                class_name = (
                    CLASS_NAMES[target_class]
                    if 0 <= target_class < len(CLASS_NAMES)
                    else f"class_{target_class}"
                )

                cells_with_heatmaps.append({
                    "bbox": cell["bbox"],
                    "heatmap": heatmap,
                    "target_class": target_class,
                    "class_name": class_name,
                    "prob": prob_target,
                    "index": idx,
                })

            st.session_state.cells_with_heatmaps = cells_with_heatmaps
            st.success("Inferência + XAI concluídos para todas as células!")

# --------------------------
# Reconstrução global com XAI + Tabela + Download
# --------------------------
if uploaded_img is not None and st.session_state.cells_with_heatmaps is not None:
    st.markdown("---")
    st.subheader("Reconstrução da imagem com XAI sobreposto")

    original_img = load_image_pil(uploaded_img)
    overlay_img = reconstruct_full_heatmap(original_img, st.session_state.cells_with_heatmaps)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Imagem original", use_container_width=True)
    with col2:
        st.image(overlay_img, caption="Imagem com XAI sobreposto", use_container_width=True)

    # -------- NOVO: botão de download da imagem --------
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="⬇️ Baixar imagem com XAI (PNG)",
        data=buf,
        file_name="xai_overlay.png",
        mime="image/png",
    )

    # -------- NOVO: tabela com resultados por célula --------
    st.subheader("Resultados por célula")

    rows = []
    for cell in st.session_state.cells_with_heatmaps:
        y0, x0, y1, x1 = cell["bbox"]
        rows.append({
            "cell_index": cell.get("index", None),
            "y0": y0,
            "x0": x0,
            "y1": y1,
            "x1": x1,
            "pred_class_id": cell["target_class"],
            "pred_class_name": cell["class_name"],
            "pred_prob": cell["prob"],
        })

    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True)
