from setuptools import setup

setup(
    name='llmconfig',
    version='0.7.2',
    python_requires='>=3.6',
    packages=['llmconfig'],
    package_dir={'llmconfig': 'llmconfig'},
    install_requires=[
        'boto3',
        'readerwriterlock',
        'gradio',
        # 'gradio==3.44.3',
    ],
    entry_points={
        'console_scripts': [
            'llmconfigcli=llmconfig.llmconfigcli:main'
            ],
    },
)
