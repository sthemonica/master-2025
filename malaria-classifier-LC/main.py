import torch
from malaria_classifier.transforms import get_transforms, get_dataloader
from malaria_classifier.models.resnet_model import ResNet50Model
from malaria_classifier.models.swin_model import SwinModel
from malaria_classifier.train import run_kfold_training

if __name__ == "__main__":
    dataset_path = "./dataset"  # ajuste esse caminho conforme seu projeto
    img_size = 224
    batch_size = 32
    num_epochs = 10
    k_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, val_tf = get_transforms(img_size)
    dataloader, class_to_idx = get_dataloader(dataset_path, train_tf, batch_size)

    # Teste com ResNet
    print("Treinando ResNet50...")
    run_kfold_training(ResNet50Model, dataloader.dataset, k_folds, batch_size, num_epochs=num_epochs, device=device)

    # Teste com Swin
    print("Treinando Swin Transformer...")
    run_kfold_training(SwinModel, dataloader.dataset, k_folds, batch_size, num_epochs=num_epochs, device=device)