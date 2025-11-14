import io
import numpy as np
import cv2
from PIL import Image, ImageDraw

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
import pandas as pd


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
# ESTILO / CSS
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #fafafa;
    }
    .pipeline-step {
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        margin-bottom: 0.8rem;
    }
    .pipeline-step h3 {
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1>🦟 Malaria XAI Pipeline</h1><h4>Cellpose ➜ Modelo ➜ XAI ➜ Reconstrução</h4>",
    unsafe_allow_html=True,
)


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
            model_type="cyto2",
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

        crops.append(
            {
                "crop": crop_224,
                "bbox": (y0, x0, y1, x1),
                "label": reg.label,
            }
        )

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
            heat_resized,
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
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "overlay_img" not in st.session_state:
    st.session_state.overlay_img = None
if "model" not in st.session_state:
    st.session_state.model = None
if "gradcam_obj" not in st.session_state:
    st.session_state.gradcam_obj = None
if "current_model_key" not in st.session_state:
    st.session_state.current_model_key = None
if "diagnosis_text" not in st.session_state:
    st.session_state.diagnosis_text = None
if "diagnosis_img" not in st.session_state:
    st.session_state.diagnosis_img = None


# =========================================================
# SIDEBAR – CONFIGURAÇÃO DA PIPELINE
# =========================================================
st.sidebar.header("⚙️ Configurações da pipeline")

uploaded_img = st.sidebar.file_uploader(
    "1) Envie uma imagem (BMP/JPG/PNG)",
    type=["bmp", "jpg", "jpeg", "png"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("2) Modelo de classificação")

model_option = st.sidebar.selectbox(
    "Escolha um modelo pré-definido",
    ["ResNet50 fold5", "Swin-T fold2"],
)

use_custom_model = st.sidebar.checkbox("Ou enviar um modelo .pth customizado")
custom_model_file = None
base_model_for_custom = None

if use_custom_model:
    custom_model_file = st.sidebar.file_uploader(
        "Envie o arquivo .pth do modelo",
        type=["pth"],
    )
    base_model_for_custom = st.sidebar.selectbox(
        "Modelo base do .pth",
        ["ResNet50", "Swin-T"],
    )

st.sidebar.markdown("---")
st.sidebar.subheader("3) Método de XAI")

xai_method = st.sidebar.selectbox(
    "Selecione o método de XAI",
    ["GradCAM", "Integrated Gradients", "Occlusion"],
)

st.sidebar.markdown("---")
run_pipeline = st.sidebar.button("🚀 Rodar pipeline completa")


# =========================================================
# MAIN – EXECUÇÃO E RESULTADOS
# =========================================================
if uploaded_img is not None:
    original_img = load_image_pil(uploaded_img)

    st.markdown(
        '<div class="pipeline-step"><h3>🖼️ Imagem de entrada</h3></div>',
        unsafe_allow_html=True,
    )
    st.image(original_img, caption="Imagem original", use_container_width=True)

    if run_pipeline:
        if uploaded_img is None:
            st.error("Por favor, envie uma imagem antes de rodar a pipeline.")
        elif use_custom_model and custom_model_file is None:
            st.error("Você marcou modelo customizado mas não enviou o arquivo .pth.")
        else:
            # Barra principal
            pipeline_bar = st.progress(0.0)
            pipeline_status = st.empty()

            # ----------------------------------------------------
            # 1) CELLPOSE
            # ----------------------------------------------------
            pipeline_status.markdown(
                "🔬 **Etapa 1/4:** Segmentando células com Cellpose..."
            )
            pipeline_bar.progress(0.10)

            cells, mask_viz = run_cellpose_and_crop(original_img)
            st.session_state.cells = cells
            st.session_state.mask_viz = mask_viz

            pipeline_bar.progress(0.25)
            pipeline_status.markdown(
                f"🔬 **Etapa 1/4 concluída:** {len(cells)} células detectadas."
            )

            # ----------------------------------------------------
            # 2) CARREGAR MODELO
            # ----------------------------------------------------
            pipeline_status.markdown(
                "🧠 **Etapa 2/4:** Carregando modelo de classificação..."
            )
            pipeline_bar.progress(0.35)

            # Define a chave interna do modelo
            if use_custom_model:
                model_key = f"custom_{base_model_for_custom}_{custom_model_file.name}"
            else:
                model_key = f"predef_{model_option}"

            # Carrega modelo apenas se mudou
            if st.session_state.current_model_key != model_key:
                if use_custom_model:
                    model, gradcam_obj = load_custom_model_from_pth(
                        custom_model_file,
                        base_model=base_model_for_custom,
                    )
                else:
                    if model_option == "ResNet50 fold5":
                        model, gradcam_obj = load_resnet_model(RESNET_CKPT_PATH)
                    else:
                        model, gradcam_obj = load_swin_model(SWIN_CKPT_PATH)

                st.session_state.model = model
                st.session_state.gradcam_obj = gradcam_obj
                st.session_state.current_model_key = model_key
            else:
                model = st.session_state.model
                gradcam_obj = st.session_state.gradcam_obj

            pipeline_bar.progress(0.45)
            pipeline_status.markdown("🧠 **Etapa 2/4 concluída:** Modelo carregado.")

            # ----------------------------------------------------
            # 3) XAI NAS CÉLULAS
            # ----------------------------------------------------
            pipeline_status.markdown(
                "🌡️ **Etapa 3/4:** Aplicando XAI nas células detectadas..."
            )
            pipeline_bar.progress(0.50)

            # Limite de células para performance
            max_cells_xai = 40
            cells_subset = cells[:max_cells_xai]

            xai_progress = st.progress(0.0)
            xai_status = st.empty()

            cells_with_heatmaps = []
            rows = []

            total = len(cells_subset)

            for i, cell in enumerate(cells_subset):
                xai_status.markdown(
                    f"🌡️ Rodando XAI na célula **{i+1}/{total}**..."
                )
                heatmap, target_class, prob_target = run_xai_heatmap(
                    model,
                    gradcam_obj,
                    tensor_from_pil_gray(cell["crop"]),
                    method=xai_method,
                )

                class_name = CLASS_NAMES[target_class]

                cells_with_heatmaps.append(
                    {
                        "bbox": cell["bbox"],
                        "heatmap": heatmap,
                        "target_class": target_class,
                        "class_name": class_name,
                        "prob": prob_target,
                        "index": i,
                    }
                )

                y0, x0, y1, x1 = cell["bbox"]
                rows.append(
                    {
                        "cell_index": i,
                        "y0": y0,
                        "x0": x0,
                        "y1": y1,
                        "x1": x1,
                        "pred_class_id": target_class,
                        "pred_class_name": class_name,
                        "pred_prob": prob_target,
                    }
                )

                xai_progress.progress((i + 1) / total)

            st.session_state.cells_with_heatmaps = cells_with_heatmaps
            st.session_state.df_results = pd.DataFrame(rows)

            pipeline_bar.progress(0.75)
            pipeline_status.markdown(
                "🌡️ **Etapa 3/4 concluída:** XAI aplicado nas células."
            )

            # ----------------------------------------------------
            # 4) RECONSTRUÇÃO
            # ----------------------------------------------------
            pipeline_status.markdown(
                "🧩 **Etapa 4/4:** Reconstruindo imagem com XAI..."
            )
            pipeline_bar.progress(0.85)

            overlay_img = reconstruct_full_heatmap(original_img, cells_with_heatmaps)
            st.session_state.overlay_img = overlay_img

            # ---- NOVO: diagnóstico da amostra com base no df_results ----
            df = st.session_state.df_results
            has_infected = df["pred_class_id"].eq(1).any()

            if has_infected:
                st.session_state.diagnosis_text = (
                    "⚠️ A amostra foi classificada como **contaminada com malária**. "
                    "Pelo menos uma célula foi identificada como infectada pelo modelo."
                )

                img_diag = original_img.convert("RGB").copy()
                draw = ImageDraw.Draw(img_diag)

                for row in df.itertuples():
                    if row.pred_class_id == 1:
                        y0, x0, y1, x1 = (
                            int(row.y0),
                            int(row.x0),
                            int(row.y1),
                            int(row.x1),
                        )
                        draw.ellipse(
                            (x0, y0, x1, y1),
                            outline="red",
                            width=4,
                        )

                st.session_state.diagnosis_img = img_diag
            else:
                st.session_state.diagnosis_text = (
                    "✅ A amostra foi classificada como **não infectada**. "
                    "Nenhuma célula foi identificada como alterada pelo modelo."
                )
                st.session_state.diagnosis_img = original_img

            pipeline_bar.progress(1.0)
            pipeline_status.markdown("✅ **Pipeline concluída com sucesso!**")


# =========================================================
# VISUALIZAÇÃO DOS RESULTADOS
# =========================================================
if st.session_state.mask_viz is not None:
    st.markdown(
        '<div class="pipeline-step"><h3>🧬 Máscara do Cellpose</h3></div>',
        unsafe_allow_html=True,
    )
    st.image(
        st.session_state.mask_viz,
        caption="Máscara gerada pelo Cellpose",
        use_container_width=True,
    )

if st.session_state.cells is not None:
    st.markdown(
        '<div class="pipeline-step"><h3>🧪 Células detectadas (crops 224×224)</h3></div>',
        unsafe_allow_html=True,
    )
    max_show = min(16, len(st.session_state.cells))
    cols = st.columns(4)
    for i in range(max_show):
        with cols[i % 4]:
            st.image(
                st.session_state.cells[i]["crop"],
                caption=f"Célula {i}",
                use_container_width=True,
            )
    if len(st.session_state.cells) > max_show:
        st.caption(
            f"... e mais {len(st.session_state.cells) - max_show} células não exibidas aqui."
        )

if st.session_state.overlay_img is not None:
    st.markdown(
        '<div class="pipeline-step"><h3>🌡️ Reconstrução com XAI sobreposto</h3></div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            load_image_pil(uploaded_img),
            caption="Imagem original",
            use_container_width=True,
        )
    with col2:
        st.image(
            st.session_state.overlay_img,
            caption="Imagem com XAI sobreposto",
            use_container_width=True,
        )

    buf_img = io.BytesIO()
    st.session_state.overlay_img.save(buf_img, format="PNG")
    buf_img.seek(0)
    st.download_button(
        label="⬇️ Baixar imagem com XAI (PNG)",
        data=buf_img,
        file_name="xai_overlay.png",
        mime="image/png",
    )

if st.session_state.df_results is not None:
    st.markdown(
        '<div class="pipeline-step"><h3>📊 Resultados por célula</h3></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(st.session_state.df_results, use_container_width=True)

    csv_bytes = st.session_state.df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar resultados em CSV",
        data=csv_bytes,
        file_name="cell_results_xai.csv",
        mime="text/csv",
    )

if st.session_state.diagnosis_text is not None:
    st.markdown(
        '<div class="pipeline-step"><h3>🧾 Conclusão da análise da amostra</h3></div>',
        unsafe_allow_html=True,
    )
    st.write(st.session_state.diagnosis_text)

    if st.session_state.diagnosis_img is not None:
        st.image(
            st.session_state.diagnosis_img,
            caption="Células destacadas conforme classificação do modelo",
            use_container_width=True,
        )
