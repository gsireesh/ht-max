from papermage import Box, CaptionsFieldName, TablesFieldName
from papermage.recipes import CoreRecipe
from papermage.recipes.recipe import Recipe
from fire import Fire


def get_nearby_captions(entity, doc, expansion_factor):
    # assuming we have one important box for the entity.
    box = entity.boxes[0]

    # we only expand out in a vertical direction because captions tend to be above or below a
    # figure/table, rather than beside.
    exp_h = expansion_factor * box.h
    diff_h = exp_h - box.h

    search_box = Box(l=box.l, t=box.t - diff_h / 2, w=box.w, h=exp_h, page=box.page)
    potential_captions = doc.find(query=search_box, name=CaptionsFieldName)
    return potential_captions


def parse_and_get_table_captions(pdf_path: str, recipe: Recipe = None) -> list[str]:
    if recipe is None:
        recipe = CoreRecipe(dpi=150)

    parsed_paper = recipe.from_pdf(pdf_path)

    all_captions = []
    for table in getattr(parsed_paper, TablesFieldName):
        all_captions.extend(get_nearby_captions(table, parsed_paper, 1.4))

    return all_captions


if __name__ == "__main__":
    Fire(parse_and_get_table_captions)
