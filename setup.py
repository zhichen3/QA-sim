from setuptools import setup, find_packages

setup(name='quasim',
      description='quantum interferometry telescope simulator',
      url='https://github.com/zhichen3/QA-sim',
      author='Zhi Chen',
      author_email='zhichen0428@gmail.com',
      license='BSD',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib','scipy','pandas','corner','uncertainties'])
