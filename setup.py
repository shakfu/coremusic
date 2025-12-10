import os

from setuptools import setup, Extension

from Cython.Build import cythonize

LIMITED_API = False
LIMITED_API_PYTHON_VERSION = 0x030A0000  # 3.10

# Ableton Link include paths
LINK_INCLUDES = [
    "thirdparty/link/include",
    "thirdparty/link/modules/asio-standalone/asio/include",
]

os.environ['LDFLAGS'] = " ".join([
    "-framework CoreServices",
    "-framework CoreFoundation",
    "-framework AudioUnit",
    "-framework AudioToolbox",
    "-framework CoreAudio",
])

DEFINE_MACROS = [
    ("LINK_PLATFORM_MACOSX", "1"),
]

if LIMITED_API:
    DEFINE_MACROS.append(
        ("Py_LIMITED_API", LIMITED_API_PYTHON_VERSION),
    )

extensions = [
    Extension(
        "coremusic.capi",
        sources=[
            "src/coremusic/capi.pyx",
        ],
        define_macros=DEFINE_MACROS,
        py_limited_api=LIMITED_API,
    ),
    Extension(
        "coremusic.link",
        sources=[
            "src/coremusic/link.pyx",
        ],
        include_dirs=LINK_INCLUDES,
        define_macros=DEFINE_MACROS,
        py_limited_api=LIMITED_API,
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]


setup(
    name="coremusic",
    description="coreaudio/coremidi/ableton-link in cython",
    ext_modules=cythonize(extensions, 
        compiler_directives={
            'language_level' : '3',
            'embedsignature': True,
        }),
    package_dir={"": "src"},
)
