## Induction Furnace

This project intends to design a general purpose drive circuit for a 48V ~5kW max induction furnace for melting metals for small scale metal casting

- Clone the `induction_furnace` repo 
- Run `git lfs install`
- Install the latest version of miniconda from [here](https://www.anaconda.com/docs/getting-started/miniconda/main#should-i-install-miniconda-or-anaconda-distribution) if not already installed.
    - The defualt configs should be best for git bash on windows: install for user only, and don't add to path (as this will add the conda executable to user PATH, while git bash looks for system PATH). 
    - Instead add the paths below manually to the system PATH environmental variable.
        - `C:\Users\<username>\miniconda3\`
        - `C:\Users\<username>\miniconda3\Scripts\`
        - `C:\Users\<username>\miniconda3\Library\bin\`
        - `C:\Users\<username>\miniconda3\Library\user\bin\`
        - `C:\Users\<username>\miniconda3\Library\mingw-w64\bin\`
    - A restart is required after adding to PATH
    - Check that it is installed with `conda --version` in a new Powershell terminal.
    - Make sure conda is updated: `conda update -n base -c defaults conda`

## Setup Conda Env
When setting up for the first time:
- create env from file: `conda env create -f IF_conda_env.yml`
- activate the env: `conda activate IF_env`
- In order for jupyter lab to recognize the virtual environment kernel, run `python -m ipykernel install --user --name=IF_env --display-name "Python (IF_env)"` from within the activated conda environment. NOTE: you may still need to select the kernel in jupyter lab after this step.

On all subsequent uses:
- activate the env: `conda activate IF_env`
- update with any changes to the yml file: `conda env update -f IF_conda_env.yml`

For a clean install:
- `conda deactivate`
- `conda remove -n IF_env --all`
- `conda env create -f IF_conda_env.yml`

## Auto Activate Conda Env in VSCode Terminal
- `IF_env` should already be set as the default conda environment in the vscode `settings.json`
```json
"terminal.integrated.env.windows": {"CONDA_DEFAULT_ENV": "IF_env"}
```
- Edit the user `.bashrc` to initialize conda every time the vscode terminal is opened
```bash
# --- [ Conda Initialization - POSIX Compatible ] -----------------------------
if [ -f "/c/Users/<user>/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/c/Users/<user>/miniconda3/etc/profile.d/conda.sh"
fi

# --- [ Auto-activate VS Code project environment if specified ] --------------
# VS Code passes CONDA_DEFAULT_ENV via settings.json
if [[ "$TERM_PROGRAM" == "vscode" && -n "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" != "$CONDA_PREFIX" ]]; then
    echo "Auto-activating conda env: $CONDA_DEFAULT_ENV"
    conda activate "$CONDA_DEFAULT_ENV"
fi
```