import torch
import timm
from PIL import Image
from torchvision import transforms


class ImageDetector:

    def __init__(self, model_path, device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=2
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

        self.class_names = ["fake","real"]


    def predict(self,image_path):

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(
            image
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence,prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_label = self.class_names[
            prediction.item()
        ]

        confidence_score = confidence.item()*100

        return {
            "prediction":predicted_label,
            "confidence":round(confidence_score,2)
        }
