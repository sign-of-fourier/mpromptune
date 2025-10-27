from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(name='m-promptune',
      version='0.2.3',
      description='DSPy acceleration',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://mpromptune.com',
      author='sign-of-fourier',
      author_email='info@quantecarlo.com',
      license='MIT',
      packages=['mpromptune'],
      zip_safe=False)


