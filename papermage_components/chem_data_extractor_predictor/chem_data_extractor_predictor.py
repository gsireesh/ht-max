from typing import List

from papermage.magelib import Metadata
import requests

from papermage_components.interfaces.token_classification_predictor import (
    TokenClassificationPredictorABC,
    EntityCharSpan,
)


class ChemDataExtractorPredictor(TokenClassificationPredictorABC):
    def __init__(self, cde_service_url):
        self.cde_service_url = cde_service_url

    def predictor_identifier(self):
        return "ChemDataExtractor"

    def tag_entities_in_batch(self, batch: List[str]) -> List[List[EntityCharSpan]]:
        req = requests.post(self.cde_service_url + "/annotate_strings", json=batch)

        if req.status_code != 200:
            raise Exception(f"Request returned status code of {req.status_code}!")
        entity_list = [
            [
                EntityCharSpan(
                    e_type=e["entity_type"],
                    start_char=e["start_char"],
                    end_char=e["end_char"],
                    metadata=Metadata(),
                )
                for e in instance
            ]
            for instance in req.json()
        ]

        return entity_list
