from setuptools import find_packages, setup

setup(
    name='ViTrainer',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/achatur18/ViTrainer',
    license='MIT',
    author='achatur18',
    author_email='abhay.chaturvedi@awone.ai',
    description='A single library for training your computer vision models.',
    install_requires=[
        'numpy',
        "torch",
        "colorama",
        "fastapi",
        "uvicorn",
        "python-multipart"
    ],
)
