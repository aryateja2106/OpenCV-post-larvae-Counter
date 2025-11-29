from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e.'

def get_requirements(file_name:str) ->List[str]:
    '''
    this function returns the requirements
    '''
    requirements=[]
    with open(file_name,'r') as file:
        requirements=file.readlines()
        requirements=[req.replace('\n',"") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="ShrimpSeedCounter",
    version='0.0.1',
    author="rishendra",
    author_email="mrishe6@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)