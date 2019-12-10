import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
                 name="superautodiff",
                 version="1.0.1",
                 author="Jovin Leong; Lucie Gillet; Huahua Zheng, Sakari Jukarainen",
                 author_email="jovinleong@g.harvard.edu",
                 description="The best automatic differentiation package for Python",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/Team-Gillet/cs207-FinalProject",
                 packages=setuptools.find_packages(),
                 install_requires=['numpy', 'pandas', 'pytest']
                 classifiers=[
                              "Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent",
                              ],
                 )

