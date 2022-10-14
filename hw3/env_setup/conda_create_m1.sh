conda remove -n "dl3"
conda env create -n "dl3" -f csci1470-m1.yml

## Install new environment.
python -m ipykernel install --user --name dl3 --display-name "DL3 (3.9)"
conda activate DL3 

## Tensorflow metal might malfunction for some students. Better to remove it
pip uninstall tensorflow-metal
