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
# SIDEBAR – CONFIGURAÇÃO DA PIPELINE
# =========================================================
st.sidebar.header("⚙️ Configurações da pipeline")

uploaded_imgs = st.sidebar.file_uploader(
    "1) Envie até 10 imagens (BMP/JPG/PNG)",
    type=["bmp", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# Limita a 10 imagens
if uploaded_imgs and len(uploaded_imgs) > 10:
    st.sidebar.warning("Você enviou mais de 10 imagens. Apenas as 10 primeiras serão processadas.")
    uploaded_imgs = uploaded_imgs[:10]

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
# MAIN – EXECUÇÃO E RESULTADOS (MULTI-IMAGENS)
# =========================================================
if uploaded_imgs:
    st.markdown(
        '<div class="pipeline-step"><h3>🖼️ Imagens enviadas</h3></div>',
        unsafe_allow_html=True,
    )

    for img in uploaded_imgs:
        st.write(f"📌 {img.name}")
        st.image(load_image_pil(img), use_container_width=True)

    if run_pipeline:

        if use_custom_model and custom_model_file is None:
            st.error("Você marcou modelo customizado mas não enviou o arquivo .pth.")
        else:
            total_imgs = len(uploaded_imgs)
            geral_progress = st.progress(0.0)
            geral_status = st.empty()

            # Carrega o modelo UMA vez
            if use_custom_model:
                model, gradcam_obj = load_custom_model_from_pth(
                    custom_model_file, base_model_for_custom
                )
            else:
                if model_option == "ResNet50 fold5":
                    model, gradcam_obj = load_resnet_model(RESNET_CKPT_PATH)
                else:
                    model, gradcam_obj = load_swin_model(SWIN_CKPT_PATH)

            df_lista = []  # lista com df de cada imagem

            # Loop principal: uma análise por imagem
            for idx_img, uploaded_img in enumerate(uploaded_imgs):

                nome_img = uploaded_img.name
                original_img = load_image_pil(uploaded_img)

                st.markdown(
                    f'<div class="pipeline-step"><h3>📄 Análise da imagem: {nome_img}</h3></div>',
                    unsafe_allow_html=True,
                )
                st.image(original_img, caption="Imagem original", use_container_width=True)

                img_bar = st.progress(0.0)
                img_status = st.empty()

                # ---------------------------
                # 1) CELLPOSE
                # ---------------------------
                img_status.markdown("🔬 **Etapa 1/4:** Segmentando células com Cellpose...")
                cells, mask_viz = run_cellpose_and_crop(original_img)
                img_bar.progress(0.25)
                img_status.markdown(f"🔬 **Etapa 1/4 concluída:** {len(cells)} células detectadas.")

                # ---------------------------
                # 2) XAI por célula
                # ---------------------------
                img_status.markdown("🌡️ **Etapa 2/4:** Aplicando XAI nas células detectadas...")
                max_cells_xai = 40
                cells_subset = cells[:max_cells_xai]

                cells_with_heatmaps = []
                rows = []

                total_cells = len(cells_subset)
                cell_prog = st.progress(0.0)

                for i, cell in enumerate(cells_subset):
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
                            "imagem": nome_img,
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

                    cell_prog.progress((i + 1) / max(1, total_cells))

                df_img = pd.DataFrame(rows)
                df_lista.append(df_img)

                img_bar.progress(0.55)
                img_status.markdown("🌡️ **Etapa 2/4 concluída:** XAI aplicado nas células.")

                # ---------------------------
                # 3) Reconstrução
                # ---------------------------
                img_status.markdown("🧩 **Etapa 3/4:** Reconstruindo imagem com XAI...")
                overlay_img = reconstruct_full_heatmap(original_img, cells_with_heatmaps)
                img_bar.progress(0.75)

                # ---------------------------
                # 4) Diagnóstico
                # ---------------------------
                has_infected = df_img["pred_class_id"].eq(1).any()

                if has_infected:
                    diagnosis_text = (
                        f"⚠️ A imagem **{nome_img}** foi classificada como "
                        f"**contaminada com malária**. Pelo menos uma célula foi "
                        f"identificada como infectada."
                    )
                    img_diag = original_img.convert("RGB").copy()
                    draw = ImageDraw.Draw(img_diag)
                    for row in df_img.itertuples():
                        if row.pred_class_id == 1:
                            draw.ellipse(
                                (row.x0, row.y0, row.x1, row.y1),
                                outline="red",
                                width=4,
                            )
                else:
                    diagnosis_text = (
                        f"✅ A imagem **{nome_img}** foi classificada como "
                        f"**não infectada**. Nenhuma célula foi identificada como alterada."
                    )
                    img_diag = original_img

                img_bar.progress(1.0)
                img_status.markdown("✅ **Pipeline dessa imagem concluída!**")

                # ---------------------------
                # VISUALIZAÇÃO POR IMAGEM
                # ---------------------------
                st.markdown("#### 🧾 Conclusão da análise")
                st.write(diagnosis_text)

                st.markdown("#### 🧬 Máscara Cellpose")
                st.image(mask_viz, use_container_width=True)

                st.markdown("#### 🌡️ Imagem com XAI sobreposto")
                st.image(overlay_img, use_container_width=True)

                st.markdown("#### 🔴 Células destacadas")
                st.image(
                    img_diag,
                    caption="Células marcadas conforme classificação do modelo",
                    use_container_width=True,
                )

                # Download imagem com XAI
                buf_img = io.BytesIO()
                overlay_img.save(buf_img, format="PNG")
                buf_img.seek(0)
                st.download_button(
                    label=f"⬇️ Baixar imagem com XAI (PNG) – {nome_img}",
                    data=buf_img,
                    file_name=f"xai_overlay_{nome_img}.png",
                    mime="image/png",
                )

                # Tabela e CSV por imagem
                st.markdown("#### 📊 Resultados por célula (imagem atual)")
                st.dataframe(df_img, use_container_width=True)

                csv_img_bytes = df_img.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Baixar CSV da imagem {nome_img}",
                    data=csv_img_bytes,
                    file_name=f"cell_results_{nome_img}.csv",
                    mime="text/csv",
                )

                # Atualiza progresso geral
                geral_progress.progress((idx_img + 1) / total_imgs)

            # =========================================================
            # CSV FINAL UNIFICADO (todas as imagens)
            # =========================================================
            if df_lista:
                df_final = pd.concat(df_lista, ignore_index=True)
                st.markdown(
                    '<div class="pipeline-step"><h3>📊 CSV consolidado das imagens</h3></div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(df_final, use_container_width=True)

                st.download_button(
                    "⬇️ Baixar CSV consolidado (todas as imagens)",
                    df_final.to_csv(index=False).encode("utf-8"),
                    file_name="resultados_multiplos_imagens.csv",
                    mime="text/csv",
                )
