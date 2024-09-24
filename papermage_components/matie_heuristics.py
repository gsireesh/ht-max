from collections import Counter
import re

from mendeleev import element
import networkx as nx
import pandas as pd
from papermage import Document, Span
import streamlit as st

composed_alloy_re = re.compile(
    "(\(?(((?P<alloy_component>[A-Z][a-z]?) (?P<fraction>0\.\d+)+?) ?)+)(\) ?[CBNO])?"
)
fraction_re = re.compile("(?P<element>[A-Z][a-z]?)+ (?P<fraction>\d\.\d+)")


def normalize_entity_string(entity_string: str):
    return entity_string.replace("-\n", "").replace("\n", " ")


def get_most_common_materials(matie_entities, n=3):
    # there's maybe an opportunity to do a little more clustering here?
    material_strings = [
        normalize_entity_string(e.text)
        for e in matie_entities
        if e.metadata["entity_type"] == "Material"
    ]
    counter = Counter(material_strings)
    return counter.most_common(n)


def get_composition_table(matie_entities, n=3):
    material_strings = [
        normalize_entity_string(e.text)
        for e in matie_entities
        if e.metadata["entity_type"] == "Material"
    ]

    composition_dict = {}
    i = 0
    for ms in material_strings:
        i + 1
        if i >= n:
            break
        if not composed_alloy_re.match(ms):
            continue
        i += 1
        composition_dict[ms] = {}
        for match in fraction_re.finditer(ms):
            composition_dict[ms][match.group("element")] = float(match.group("fraction"))

    composition_df = pd.DataFrame(composition_dict).T
    composition_df = composition_df.reindex(
        sorted(composition_df.columns, key=lambda x: element(x).atomic_number), axis=1
    )
    return composition_df.fillna(0)


@st.cache_data
def create_document_graph(_doc: Document, identifier: str = None) -> nx.Graph:
    doc_graph = nx.Graph()
    for section in _doc.reading_order_sections:
        clean_section_name = section.metadata["section_name"].replace(" ", "_")
        section_prefix = f"{clean_section_name}_{section.metadata['paragraph_reading_order']}"
        for entity in section.TAGGED_ENTITIES_MatIE:
            doc_graph.add_node(
                section_prefix + "_" + entity.metadata["entity_id"],
                entity_type=entity.metadata["entity_type"],
                entity_text=entity.text,
                entity_section=section.metadata["section_name"],
            )

        for relation in section.metadata["in_section_relations"]:
            node1 = section_prefix + "_" + relation["arg1"]
            node2 = section_prefix + "_" + relation["arg2"]
            if node1 not in doc_graph or node2 not in doc_graph:
                print(node1, node2)
                continue
            doc_graph.add_edge(node1, node2, relation_type=relation["relation_type"])

    return doc_graph


def gnp(graph, node, property):
    if not node:
        return None
    return normalize_entity_string(graph.nodes[node][property])


def get_neighbors_of_type(graph, node, e_type):
    return [n for n in graph.neighbors(node) if graph.nodes[n]["entity_type"] == e_type]


def get_property_table(doc_graph):
    property_table = []
    property_nodes = [
        node for node in doc_graph if doc_graph.nodes[node]["entity_type"] in ["Property"]
    ]
    property_strings = [gnp(doc_graph, node, "entity_text") for node in property_nodes]
    for property_node, property_string in zip(property_nodes, property_strings):
        material_neighbors = get_neighbors_of_type(doc_graph, property_node, "Material")
        result_neighbors = get_neighbors_of_type(doc_graph, property_node, "Result")
        for mat_node in material_neighbors:
            for result_node in result_neighbors:
                property_table.append(
                    {
                        "material": gnp(doc_graph, mat_node, "entity_text"),
                        "property": property_string,
                        "result": gnp(doc_graph, result_node, "entity_text"),
                        "section": gnp(doc_graph, mat_node, "entity_section"),
                    }
                )
    return pd.DataFrame(property_table)


def get_synthesis_method_table(doc_graph):
    property_table = []
    synthesis_nodes = [
        node for node in doc_graph if doc_graph.nodes[node]["entity_type"] in ["Synthesis"]
    ]
    synthesis_strings = [gnp(doc_graph, node, "entity_text") for node in synthesis_nodes]
    for synthesis_node, synthesis_string in zip(synthesis_nodes, synthesis_strings):
        env_neighbors = get_neighbors_of_type(doc_graph, synthesis_node, "Environment")
        amt_neighbors = get_neighbors_of_type(doc_graph, synthesis_node, "Amount_Unit")
        for env_neighbor in env_neighbors or [None]:
            for amt_neighbor in amt_neighbors or [None]:
                amt_value_nodes = (
                    get_neighbors_of_type(doc_graph, amt_neighbor, "Number")
                    if amt_neighbor
                    else None
                )
                amt_value_node = None if not amt_value_nodes else amt_value_nodes[0]

                env_unit_nodes = (
                    get_neighbors_of_type(doc_graph, env_neighbor, "Amount_Unit")
                    if env_neighbor
                    else None
                )
                env_unit_node = env_unit_nodes[0] if env_unit_nodes else None
                env_amt_nodes = (
                    get_neighbors_of_type(doc_graph, env_unit_node, "Number")
                    if env_unit_node
                    else None
                )
                env_amt_node = env_amt_nodes[0] if env_amt_nodes else None

                property_table.append(
                    {
                        "synthesis_method": synthesis_string,
                        "amount_value": gnp(doc_graph, amt_value_node, "entity_text"),
                        "amount_unit": gnp(doc_graph, amt_neighbor, "entity_text"),
                        "environment": gnp(doc_graph, env_neighbor, "entity_text"),
                        "environment_value": gnp(doc_graph, env_amt_node, "entity_text"),
                        "environment_unit": gnp(doc_graph, env_unit_node, "entity_text"),
                        "section": gnp(doc_graph, synthesis_node, "entity_section"),
                    }
                )
    return pd.DataFrame(property_table)
