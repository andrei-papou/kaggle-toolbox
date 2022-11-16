import os
from typing import List

import setuptools
from urllib.parse import parse_qsl

GITHUB_TOKEN_ENV = 'GITHUB_TOKEN'
GITHUB_TOKEN_VAR1 = '${' + GITHUB_TOKEN_ENV + '}'
GITHUB_TOKEN_VAR2 = '$' + GITHUB_TOKEN_ENV


def read_requirements(req_file_path: str) -> List[str]:
    token = os.environ.get(GITHUB_TOKEN_ENV, '')
    requirement_url_list = []
    with open(req_file_path) as req_file:
        for s in req_file.readlines():
            if not s or s.startswith('-f '):
                continue
            requirement_url = s.strip().replace(GITHUB_TOKEN_VAR1, token).replace(GITHUB_TOKEN_VAR2, token)
            if '#' in requirement_url:
                _, query = requirement_url.split('#')
                query_dict = dict(parse_qsl(query))
                requirement_url_list.append(query_dict['egg'] + ' @ ' + requirement_url)
            else:
                requirement_url_list.append(requirement_url)
        return requirement_url_list


with open('README.md', 'r') as f:
    long_description = f.read()

core_requirement_list = read_requirements('requirements/base/core.txt')
laptop_requirement_list = read_requirements('requirements/base/laptop.txt')
remote_requirement_list = read_requirements('requirements/base/remote.txt')

logger_tensorboard_requirement_list = read_requirements('requirements/base/loggers/tensorboard.txt')
logger_wandb_requirement_list = read_requirements('requirements/base/loggers/wandb.txt')

requirement_list = core_requirement_list

setuptools.setup(
    name='kaggle_toolbox',
    version='0.1.6',
    description='Toolbox library for Kaggle competitions.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/andrei-papou/kaggle-toolbox',
    packages=setuptools.find_packages(include='kaggle_toolbox/*'),
    install_requires=requirement_list,
    extras_require={
        'local': laptop_requirement_list,
        'remote': remote_requirement_list,
        'tensorboard': logger_tensorboard_requirement_list,
        'wandb': logger_wandb_requirement_list,
    },
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
)
