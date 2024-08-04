# ht-max

Code for the Collage Tool, a part of the HT-MAX project.


![](diagrams/fig1.svg)


Collage is a tool designed for rapid prototyping, visualization, and evaluation of different 
information extraction models on scientific PDFs. Further, we enable both non-technical users
and NLP practitioners to inspect, debug, and better understand modeling pipelines by providing 
granular views of intermediate states of processing.

This demo should be available and running at [this URL](http://windhoek.sp.cs.cmu.edu:8501). This
server can sometimes be unstable. If it is having issues when you try to access it, please
follow the Docker Compose instructions below.

## Setup/Running the Demo

For convenience, we've Dockerized all of the components of this system. To get started with the demo,
simply run:

```commandline
docker compose up
```

In the root directory of the repo. On our machines, this takes ~20 minutes to complete, largely
because of ChemDataExtractor having to download a number of models. If you do not need 
ChemDataExtractor, of want to speed uip the build process significantly, comment out the 
`chemdataextractor` service from `compose.yaml`. This sets up a Docker Compose network with three 
containers: the interface, an instance of GROBID, to get reading order sections, and 

## What's in this repo?

Collage has three primary components:

- A [PaperMage](https://github.com/allenai/papermage) backbone that underlies our PDF processing, 
defined in `papermage_components/Materials_Recipe.py`
- Three software interfaces to accelerate the rapid prototyping of different kinds of models. These
interfaces are designed around token classification, i.e. classic information extraction models, 
text-to-text models, such as LLMs, and multimodal models to process things like tables. These 
interfaces are defined in `papermage_components/interfaces`. 
- A frontend, built in streamlit, that automatically visualizes modeling results produced by those 
interfaces. The landing interface, where users can upload papers and customize the processing they 
run on them, is in `Upload_Paper.py`. The three other interface views are defined in the `pages/`
package.

### Extending Collage by implementing interfaces

This repo contains the interfaces discussed above, along with several implementations of those
repositories. These implementations provide the blueprint for how to implement the interfaces in a 
number of different ways, including in-memory implementations right in the pipeline, small, 
Dockerized services for components with complicated environment requirements that may not be 
compatible with Collage, as well as a few that use external APIs. We outline these components,
and how they implement their interface below.

[TK]


### Scripts

This repo contains the following scripts:

`parse_papers_to_json.py`: The script parses the content from PDFs into structured representations 
in json. Currently, it runs the `MaterialsRecipe` on a specified folder of papers, and dumps the json
representations to the specified output folder.

### Notebooks

To aid development, this repo contains two notebooks that facilitate quicker development of 
PaperMage predictors. `dev_run_recipe_and_serialize.ipynb` takes a new PDF, runs that 
`MaterialsRecipe` on it, and serializes the result. `dev_run_recipe_and_serialize` opens a paper 
from the parsed json, and allows further manipulation.


## [CMU Collaborators] Getting and using data

The testing data for this project is managed and versioned by [DVC](https://dvc.org), and it is stored in
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