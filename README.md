# March Machine Learning Mania 2026
This repository contains our solution for the March Machine Learning Mania 2026 Kaggle competition.

# Prerequisites
You will need the following dependencies and prerequisites
- Pixi
- Kaggle Account

## Setup
Clone repo
```
git clone xxx
cd xxx
```

### Set Up Kaggle API Credentials
You need a Kaggle API token to download the competition data via scripts (or you can download it manually)

Generate a Kaggle API token from your account settings:
- https://www.kaggle.com/settings

Follow the official Kaggle CLI setup guide (optional but helpful):
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md

Create .env file and store api key there.
This will differ depending on the shell you use. 

If you are using NuShell you can structure like this
```python
# .env file
$env.KAGGLE_API_TOKEN = "xxxxxxxxxxxxxx"
```

### Install Dependencies & Activate Environment
```
pixi install
pixi shell
source .env
```

### Register the Pixi Jupyter Kernel
Run once to make the Pixi environment visible as a notebook kernel in VS Code and Jupyter:
```
pixi run register_notebook_kernel
```
Then select `Python (MM2026 Pixi)` as the kernel when opening a notebook.

### Ingest Competition Data
```
pixi run ingest_data_into_landing
```


## Adding new data...
If you want to add new data please DO NOT commit to the repo.

Instead, use the following steps...
1. Create a pixi task in pixi.toml to ingest the data via cli or create a script that can ingest the data.
2. Add instructions on how to obtain relevant api keys for the data to the README.