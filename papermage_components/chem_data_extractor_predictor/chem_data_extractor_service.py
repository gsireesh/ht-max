from typing import List

from chemdataextractor import Document
import fastapi
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class EntityCharSpanResponse(BaseModel):
    text: str
    start_char: int
    end_char: int
    entity_type: str


@app.get("/")
def get_root():
    return {"hello": "world"}


@app.get("/status")
def get_status():
    return "Service is up!"


@app.post("/annotate_strings")
def annotate_strings(strings: List[str]) -> List[List[EntityCharSpanResponse]]:
    document = Document(*strings)

    # compute all mentions before iterating through them
    all_cems = document.cems

    all_entities = []
    for element in document.elements:
        element_entities = []
        for entity in element.cems:
            entity = EntityCharSpanResponse(text=entity.text, start_char=entity.start,
                                            end_char=entity.end_char, entity_type="CDE_Chemical")
            element_entities.append(entity)
        all_entities.append(element_entities)
    return all_entities
