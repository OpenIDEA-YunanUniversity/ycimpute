# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

from setuptools import setup, find_packages

package_name = 'ycimpute'

readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_filename, 'r') as f:
        readme = f.read()
except:
    logging.warning("Failed to load %s" % readme_filename)
    readme = ""

try:
    import pypandoc
    readme = pypandoc.convert(readme, to='rst', format='md')
except:
    logging.warning("Conversion of long_description from MD to RST failed")
    pass

if __name__ == '__main__':
    setup(
        name=package_name,
        version="0.2",
        description="Matrix completion and feature imputation algorithms",
        author="zhouyc",
        author_email="yuanchenzhouhcmy@gmail.com",
        url="https://github.com/OpenIDEA-YunanUniversity/ycimpute",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Missing Value Imputation',
        ],
        install_requires=[
            'six',
            'numpy>=1.10',
            'scipy',
            'scikit-learn>=0.17.1',
            'torch>=1.1.0',
        ],
        long_description=readme,
        packages=find_packages(),
    )
