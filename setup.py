from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name= "EXCERML",
    version= "0.0.1",
    author= "Subhraneel Haldar",
    description= "End-To_End ML Pipeline Project about Bodyweight Prediction",
    author_email= "subhraneel777@gmail.com",
    install_reqires= get_requirements("requirements.txt"),
    packages= find_packages()
)    