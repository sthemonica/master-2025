from transformers import AutoImageProcessor, SwinForImageClassification
import torch.nn as nn

class SwinModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinModel, self).__init__()
        self.model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x).logits