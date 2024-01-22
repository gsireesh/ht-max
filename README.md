# ht-max
Code for the HT-MAX project

## Setup

First, create a new conda environment and install [Papermage](https://github.com/allenai/papermage/tree/main?tab=readme-ov-file#setup).

Then, install the additional requirements:
```
pip install -r requirements.txt
```

### Getting and using data

The data for this project is managed and versioned by [DVC](https://dvc.org), and it is stored in [this Google Drive folder](https://drive.google.com/drive/u/0/folders/1XNbshzrpG01caal8ftSpF3WOrlUU2y7G). Data and checkpoints should be stored in the `data/` folder. For this project, we are symlinking in the PDF data that we store in the [NLP Collaboration Box Folder](https://cmu.app.box.com/folder/189367159764?s=8mi0zv3qbo4hjiun36y87c2vxs2y0l08), e.g.:
```bash
ln -s $BOX_SYNC_FOLDER/NLP-collaboration-folder/AM_Creep_Papers data/AM_Creep_Papers
```

Data derived from those PDFs, model checkpoints, etc. will be stored in the `data/` folder and managed with DVC.

You can find instructions for installing DVC [here](https://dvc.org/doc/install). Once you have DVC installed, run `dvc pull` from the root of the repo. This will pull down all the files that have been checked into DVC thus far. This will ask for permission for DVC to access the files in your Google Drive; you should proceed with your CMU account. 

DVC works in a similar fashion to [git-lfs](https://git-lfs.github.com/):
it stores pointers and metadata for your data in the git repository,
while the files live elsewhere (in this case, on Google Drive). As you
work with data, such as in [the DVC tutorial](https://dvc.org/doc/start/data-and-model-versioning), DVC will automatically add the files you have tracked with it to the `.gitignore` file, and add new `.dvc` files that track the metadata associated with those files.

### Sample Workflow

* **Pull data down** : run `dvc pull` to pull down the data file into the repository folder
* **Modify your data** : as you would without DVC, use, modify, and work with your data.
* **Add new/modified data to DVC** : using `dvc add ...` in a similar fashion to a `git add`, add your new or modified data files to DVC
* **Add the corresponding metadata to git** : Once the data file has been added to DVC, a corresponding `.dvc` file will have been created. Add or update this into git, then push.
* **Sync the locally updated DVC data with the remote** : finally, push the data itself up to Google Drive with the `dvc push` command.

tl;dr:

* dvc pull
* dvc add <data_file>
* git add/commit <data_file.dvc>
* git push
* dvc push

## Scripts

This repo contains the following scripts:

`parse_papers_to_json.py`: The script parses the content from PDFs into plaintext representations in json. Currently, it only extracts title, abstract, and all sentences from the paper into a json object with the following form. Section text is currently empty.

```json
{
    "title": "",
    "abstract": "",
    "sentences": [""],
    "section_text": {}
}
```
