import torch
from torchvision import models, transforms
from PIL import Image

class SatelliteFireDetector:
    def __init__(self, model_path):
        self.model = models.resnet18()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            out = self.model(img)
            prob = torch.softmax(out, dim=1)

        return {
            "no_fire": float(prob[0][0]),
            "wildfire": float(prob[0][1])
        }
