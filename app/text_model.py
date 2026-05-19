import torch
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification


class TextDetector:

    def __init__(self, model_path, device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            ignore_mismatched_sizes=True
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=False
        )

        print("Missing:", missing)
        print("Unexpected:", unexpected)

        self.model.to(self.device)

        self.model.eval()

        self.label_map = {
            0:"real",
            1:"suspicious"
        }


    def predict(self,text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        inputs = {
            k:v.to(self.device)
            for k,v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            confidence,prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_label = self.label_map[
            prediction.item()
        ]

        confidence_score = confidence.item()*100

        return {
            "prediction":predicted_label,
            "confidence":round(confidence_score,2)
        }
