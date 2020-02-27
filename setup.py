import setuptools

setuptools.setup(name='birdvoxpaint',
                 version='0.2.0-dev0',
                 description='Bird Vox False-Color Spectrograms',
                 long_description=open('README.md').read().strip(),
                 author='Phincho Sherpa, Vincent Lostanlen, Bea Steers',
                 author_email='vincent.lostanlen@nyu.edu',
                 # url='http://path-to-my-packagename',
                 packages=setuptools.find_packages(),
                 # py_modules=['packagename'],
                 # package_data={'asdf': {'*.yaml'}},
                 # include_package_data=True,
                 install_requires=[
                    'librosa>=0.7.0',
                    'matplotlib',
                    'tqdm',
                 ],
                 entry_points={
                        # 'console_scripts': [
                        #     'bvpaint = bvpaint.cli:main'
                        # ],
                 },
                 license='MIT License',
                 zip_safe=True,
                 keywords=('acoustic detection bird calls fault identification '
                           'false color spectrogram librosa'))
