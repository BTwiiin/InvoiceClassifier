from setuptools import setup, find_packages

setup(
    name='invoice_classifier',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'easyocr',
        'joblib',
        'scikit-learn',
        'pandas',
        'json'
    ],
    description='A package for invoice classification with QR code detection and text extraction',
    author='Ulad Shuhayeu',
    author_email='v.shugaev03@gmail.com',
    license='MIT',
    url='',
)
