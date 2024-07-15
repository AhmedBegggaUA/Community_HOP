# Community_HOP
Community-Hop: Enhancing Node Classification through Community Preference
## Dependencies

Conda environment
```
conda create --name <env> --file requirements.txt
```

or

```
conda env create -f conda_graphy_environment.yml
conda activate graphy
```
## Code organization
* `data/`: folder with the datasets.
* `logs/`: folder with the logs of the experiments.
* `runners/`: folder with the scripts for running the experiments.
* `splits/`: splits that we used, taking from GEO-GCN repository.
* `main.py`: script with inline arguments for running the experiments.
* `models.py`: script with our proposed architecture.
* `utils.py`: extra functions used for the experiments.