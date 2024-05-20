# ht-max
Code for the HT-MAX project

## Repository Structure

This repository contains code used to process PDFs into structured data using PaperMage's 
framework, code for the Collage frontend, and images for the ACL '24 demo paper submission about 
Collage. Subfolders' purpose are enumerated below:

`data`: The directory where data should be located. The root directory for all 
[DVC](https://dvc.org) metadata files, and data, when pulled from the Google Drive folder, will be 
located here. This allows hardcoding of paths in the `data/` directory.

`diagrams`: Diagrams and screenshots used for the ACL '24 Demo submission.

`pages`: The Collage application is designed as a Streamlit [multipage app](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app).
This folder contains the annotations and inspection view, respectively. The main streamlit app is 
defind in `1_Summary_View.py`, and can be run with `streamlit run 1_Summary_View.py` after
completing the setup below.

`papermage_components`: This repository defines a number of components inside the PaperMage framework
that allow us to use more processing than is by default available in the PaperMage distribution. 

## Setup/Running the Demo

First, create a new conda environment and install our fork of 
[Papermage](https://github.com/gsireesh/papermage/tree/ad_hoc_fixes?tab=readme-ov-file#setup), 
on the branch `ad_hoc_fixes`. We have made a few modifications to PaperMage that are necessary for the 
demo to work correctly: allowing overlapping boxes/spans in entities, and serializing images.

Then, install the additional requirements:
```
pip install -r requirements.txt
```

Then, run the demo with
```
streamlit run 1_Summary_View.py
```
The demo expects parsed paper json to be in a folder called `data/Midyear_Review_Papers_Parsed`.
This location ca be changed by changing the value of the `PARSED_PAPER_FOLDER` variable in the 
`shared_utils.py` module. Papers can be parsed using the `parse_papers_to_json.py` script outlined 
below. For anyone external to CMU trying to run this, you will need to disable the MatIE predictor, 
which relies on an unpublished model and codebase, as well as likely the Highlight Parser, which 
renders the pipeline brittle.

## Processing Pipeline

This paper defines a new PaperMage "recipe" for processing papers. A recipe is collection of 
processing steps that applies a series of models in sequence, resulting in a structured 
representation of the content of a PDF. This recipe contains steps that are CMU-specific, and 
are either specific enough to not warrant general release, or rely on unpublished assets. However, 
there are also parts that can run inside papermage without issue, which we intend to contribute back 
to the original repo.  We detail the parts of our processing pipeline that differ from the default
papermage pipeline below. All modules are contained within the `papermage_components` package, and 
the recipe itself is defined in `materials_recipe.py`

- **Grobid Reading Order Parser**, defined in `reading_order_parser.py`: This parser uses Grobid to
parse out sections and their paragraphs in the PDF, along with their number, if present. This was 
built because while Papermage can identify paragraphs and section titles, it does not associate 
titles with the text in that section. This parser produces a list of "paragraph" entities, each of 
which has metadata specifying its section and location on the page; it allows the demo to filter by 
paper section.

- **Highlight Parser**, defined in `highlightParser.py`. This parser uses PyMuPDF to extract
user highlights from a PDF and create PaperMage entities from them. Because of the way highlighting
in Adobe Acrobat works, with large, overlapping boxes, this parser tends to work badly with
PaperMage's strong assumption that all of a given entity's bounding boxes are disjoint, and should 
probably be disabled for any use of this recipe.

- **Sentence Prediction**, defined in `scispacy_sentence_predictor.py` We implement a sentence
predictor based on SciSpacy. This works better for materials science text than the default sentence
segmenter built into PaperMage, which is based on a PySBD model trained on internet text.

- **MatIE Predictor**, defined in `matIE_predictor.py` This model applies the yet-unpublished MatIE 
model to perform materials science-specific information extraction. This in particular depends on 
non-public code, and should be disabled in any use of this pipeline. 

- **Table Structure Predictor**, defined in `table_structure_predictor.py`. This predictor uses the 
Microsoft `TableTransformer` to predict the structure of tables, in order to render their contents
into a structured format.

An example of instantiating and running the Materials Recipe can be found in the notebook 
`dev_run_recipe_and_serialize.py`


### Getting and using data

The data for this project is managed and versioned by [DVC](https://dvc.org), and it is stored in
[this Google Drive folder](https://drive.google.com/drive/u/0/folders/1XNbshzrpG01caal8ftSpF3WOrlUU2y7G).
Data and checkpoints should be stored in the `data/` folder. For this project, we are symlinking 
in the PDF data that we store in the [NLP Collaboration Box Folder](https://cmu.app.box.com/folder/189367159764?s=8mi0zv3qbo4hjiun36y87c2vxs2y0l08), e.g.:

```bash
ln -s $BOX_SYNC_FOLDER/NLP-collaboration-folder/AM_Creep_Papers data/AM_Creep_Papers
```

Data derived from those PDFs, model checkpoints, etc. will be stored in the `data/` folder and 
managed with DVC.

You can find instructions for installing DVC [here](https://dvc.org/doc/install). Once you have DVC installed, run 
`dvc pull` from the root of the repo. This will pull down all the files that have been checked into 
DVC thus far. This will ask for permission for DVC to access the files in your Google Drive; 
you should proceed with your CMU account. 

DVC works in a similar fashion to [git-lfs](https://git-lfs.github.com/):
it stores pointers and metadata for your data in the git repository,
while the files live elsewhere (in this case, on Google Drive). As you
work with data, such as in [the DVC tutorial](https://dvc.org/doc/start/data-and-model-versioning), DVC will automatically add the files you have 
tracked with it to the `.gitignore` file, and add new `.dvc` files that track the metadata associated
with those files.

### Sample Workflow

* **Pull data down** : run `dvc pull` to pull down the data file into the repository folder
* **Modify your data** : as you would without DVC, use, modify, and work with your data.
* **Add new/modified data to DVC** : using `dvc add ...` in a similar fashion to a `git add`, add 
your new or modified data files to DVC
* **Add the corresponding metadata to git** : Once the data file has been added to DVC, a 
corresponding `.dvc` file will have been created. Add or update this into git, then push.
* **Sync the locally updated DVC data with the remote** : finally, push the data itself up to Google 
Drive with the `dvc push` command.

tl;dr:

* dvc pull
* dvc add <data_file>
* git add/commit <data_file.dvc>
* git push
* dvc push

## Scripts

This repo contains the following scripts:

`parse_papers_to_json.py`: The script parses the content from PDFs into structured representations 
in json. Currently, it runs the `MaterialsRecipe` on a specified folder of papers, and dumps the json
representations to the specified output folder.


lorem ipsum
