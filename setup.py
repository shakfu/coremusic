import os
# from distutils.core import setup
# from distutils.extension import Extension
from setuptools import setup, Extension


from Cython.Build import cythonize

os.environ['LDFLAGS'] = " ".join([
        "-framework CoreServices",
        "-framework CoreFoundation",
        "-framework AudioUnit",
        "-framework AudioToolbox",
        "-framework CoreAudio",
])

extensions = [
    Extension("coreaudio", ["coreaudio.pyx"],
        # define_macros = [
        #     ('PD', 1),
        # ],
        include_dirs=[
            # "../libpd_wrapper",
        ],
        libraries = [
            'm',
            'dl',
            'pthread',
            # 'portaudio', # requires portaudio to be installed system-wide
        ],
        library_dirs=[],
        extra_objects=[
        ],
    ),
]


setup(
    name="coreaudio in cython",
    ext_modules=cythonize(extensions, 
        compiler_directives={
            'language_level' : '3',
            'embedsignature': True,
        }),
)
