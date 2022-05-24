from typing import Any
import torch
from transformers import TensorType
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.generic import PaddingStrategy
from config import Config
from elg import FlaskService
from elg.model import ClassificationResponse


class PoliticalBiasService(FlaskService):

    def convert_outputs(self, content: str) -> ClassificationResponse:
        bias_dict: dict[str, float] = predict(content)
        return ClassificationResponse(classes=[{"class": k, "score": v} for k, v in bias_dict.items()])

    def process_text(self, content: Any) -> ClassificationResponse:
        return self.convert_outputs(content.content)


def predict(text: str) -> dict[str, float]:
    inputs = Config.tokenizer(text, truncation=True, padding=PaddingStrategy.MAX_LENGTH,
                              max_length=512, return_tensors=TensorType.PYTORCH)
    # first dimension is Linke/socialist, second one is FDP/liberal
    result_socioeconomic: SequenceClassifierOutput = Config.model_socioeconomic(**inputs)
    # first dimension is Green/open, second one is AfD/conservative
    result_cultural: SequenceClassifierOutput = Config.model_cultural(**inputs)
    value_socioeconomic: float = float(torch.sigmoid(result_socioeconomic.logits[0][1]))
    value_cultural: float = float(torch.sigmoid(result_cultural.logits[0][1]))
    return dict(socioeconomic=value_socioeconomic, cultural=value_cultural)


pbs: PoliticalBiasService = PoliticalBiasService(
    Config.LANGUAGE_SERVICE, path=f"/process/{Config.DOCKER_COMPOSE_SERVICE_NAME}")
app = pbs.app
