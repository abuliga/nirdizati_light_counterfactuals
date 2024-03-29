from setuptools import setup, find_packages

setup(
    name='nirdizati_light',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'dice_ml @ git+https://github.com/abuliga/DiCE.git@origin/main',
        'declare4py @ git+https://github.com/abuliga/declare4py.git@main',
        'pymining==0.2',
        'pandas~=1.5.3',
        'pm4py~=2.2.21',
        'scikit-learn~=1.3.0',
        #'shap~=0.41.0',
        'numpy~=1.22.0',
        'hyperopt~=0.2.7',
        #'tensorflow~=2.13.0',
        'dateparser~=1.1.8',
        'holidays~=0.28',
        'funcy~=2.0.0',
        'xgboost~=1.7.6',
        #'pdpbox~=0.2.1',
        'pymoo~=0.6.0'
    ]
)
