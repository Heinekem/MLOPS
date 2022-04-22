Create env
'''bash
conda create -n wineq python=3.7 -y
'''
activate env
'''bash
conda activate wineq
'''
create a requirement.txt and install it
'''bash
pip install -r requirements.txt
'''
git init

dvc init

dvc add data_given\winequality.csv

add tox and pytest

tox command
'''bash
tox
'''
for rebuilding
'''bash
tox -r 
'''
pytest command
'''bash
pytest -v
'''
Local package install command
'''bash
pip install -e .
'''
 build your own package command
 '''bash
 python setup.py sdist bdist_wheel
 '''
