from setuptools import setup, find_packages

setup(
    name='e_cars_sentiments',
    version='0.1.0',
    description='A project to analyze sentiment on electric cars in different countries using text mining and neural networks',
    author='Ajay Singh Pundir',
    author_email='ajaysinghpundir70@gmail.com',
    packages=find_packages(),
    install_requires=[
        'requests',
        'beautifulsoup4',
        'pandas',
        'jieba',
        'nltk',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'configparser'
    ],
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.6',
)
