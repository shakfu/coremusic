import os

from setuptools import setup, Extension

from Cython.Build import cythonize

LIMITED_API = False
LIMITED_API_PYTHON_VERSION = 0x030A0000 # 3.10


os.environ['LDFLAGS'] = " ".join([
        "-framework CoreServices",
        "-framework CoreFoundation",
        "-framework AudioUnit",
        "-framework AudioToolbox",
        "-framework CoreAudio",
])

DEFINE_MACROS = []

if LIMITED_API:
    DEFINE_MACROS.append(
        ("Py_LIMITED_API", LIMITED_API_PYTHON_VERSION),
    )

extensions = [
    Extension("coremusic.capi",
        sources=[
            "src/coremusic/capi.pyx",
        ],
        define_macros=DEFINE_MACROS,
        py_limited_api=LIMITED_API,
    ),
]


setup(
    name="coremusic",
    description="coreaudio/coremidi in cython",
    version="0.1.2",
    ext_modules=cythonize(extensions, 
        compiler_directives={
            'language_level' : '3',
            'embedsignature': True,
        }),
    package_dir={"": "src"},
)
