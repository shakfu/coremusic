import os

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
    Extension("coreaudio.capi", 
        sources=[
            "src/coreaudio/capi.pyx",
            "src/coreaudio/audio_player.c"
        ],
        # define_macros = [
        #     ('PD', 1),
        # ],

        # include_dirs=["."],

        # libraries = [
        #     'm',
        #     'dl',
        #     'pthread',

        # ],

        # library_dirs=[],

        # extra_objects=[],
    ),
]


setup(
    name="coreaudio in cython",
    ext_modules=cythonize(extensions, 
        compiler_directives={
            'language_level' : '3',
            'embedsignature': True,
        }),
    package_dir={"": "src"},
)
