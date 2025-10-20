# coremusic Documentation

This directory contains the Sphinx documentation for coremusic.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs-requirements.txt
```

Or using uv:

```bash
uv pip install sphinx sphinx-rtd-theme
```

### Build HTML Documentation

From the project root:

```bash
make docs
```

Or manually:

```bash
cd sphinx
sphinx-build -b html . _build/html
```

The HTML documentation will be built in `sphinx/_build/html/index.html`.

### View Documentation

Open the built documentation in your browser:

```bash
open sphinx/_build/html/index.html
```

Or serve it locally:

```bash
make docs-serve
# Visit http://localhost:8000
```

### Build PDF Documentation

```bash
make docs-pdf
```

The PDF will be created at `sphinx/_build/latex/coremusic.pdf`.

### Clean Documentation Build

```bash
make docs-clean
```

## Documentation Structure

```
sphinx/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── getting_started.rst    # Getting started guide
│
├── api/                   # API Reference
│   ├── index.rst
│   ├── audio_file.rst
│   ├── audio_unit.rst
│   └── ...
│
├── tutorials/             # Step-by-step tutorials
│   ├── index.rst
│   ├── audio_file_basics.rst
│   └── ...
│
├── cookbook/             # Recipe collection
│   ├── index.rst
│   ├── file_operations.rst
│   └── ...
│
└── examples/             # Example documentation
    ├── index.rst
    └── ...
```

## Documentation Sections

### Getting Started
Basic installation, configuration, and first steps with coremusic.

### API Reference
Complete API documentation auto-generated from docstrings:
- AudioFile and AudioFormat classes
- AudioUnit classes and functions
- AudioQueue operations
- MIDI functions and classes
- Utility functions

### Tutorials
Step-by-step tutorials covering:
- Audio file operations
- Real-time audio processing
- AudioUnit development
- MIDI processing
- Advanced techniques

### Cookbook
Ready-to-use recipes for common tasks:
- File operations
- Audio processing
- Real-time audio
- MIDI processing
- Integration with NumPy/SciPy

### Examples
Complete, working example applications demonstrating coremusic capabilities.

## Writing Documentation

### reStructuredText (RST) Basics

Sphinx uses reStructuredText format:

**Headers:**
```rst
Chapter
=======

Section
-------

Subsection
^^^^^^^^^^
```

**Code blocks:**
```rst
.. code-block:: python

   import coremusic as cm
   with cm.AudioFile("audio.wav") as audio:
       print(audio.duration)
```

**Links:**
```rst
:doc:`other_page`
:ref:`section-label`
```

**Notes and warnings:**
```rst
.. note::
   This is a note.

.. warning::
   This is a warning.
```

### Adding API Documentation

API documentation is auto-generated from docstrings using Napoleon:

```python
def my_function(param1, param2):
    """
    Brief description.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Example:
        >>> my_function("hello", 42)
        "result"
    """
    pass
```

Then reference in RST:

```rst
.. autofunction:: coremusic.my_function
```

### Adding a Tutorial

1. Create a new `.rst` file in `tutorials/`
2. Add it to `tutorials/index.rst` toctree
3. Write using RST format
4. Rebuild documentation

Example structure:

```rst
Tutorial Title
==============

Brief introduction.

Prerequisites
-------------

What the reader needs to know.

Step 1: First Task
------------------

Explanation and code:

.. code-block:: python

   import coremusic as cm
   # Code example

Step 2: Next Task
-----------------

Continue...

Complete Example
----------------

Full working code.

Next Steps
----------

Where to go from here.
```

### Adding a Recipe

1. Create or edit a file in `cookbook/`
2. Follow the recipe pattern:
   - Recipe title
   - Brief description
   - Code example
   - Usage notes

Example:

```rst
Recipe Name
^^^^^^^^^^^

Brief description of what this recipe does.

.. code-block:: python

   import coremusic as cm

   def recipe_function():
       """Recipe implementation."""
       pass

   # Usage
   recipe_function()

**Notes:**

- Important detail 1
- Important detail 2
```

## Documentation Standards

### Style Guide

- **Clear and concise**: Get to the point quickly
- **Practical examples**: Include working code
- **Error handling**: Show proper error handling
- **Platform notes**: Mention macOS-specific behavior
- **Cross-references**: Link to related content

### Code Examples

- **Complete**: Can be run as-is
- **Practical**: Solve real problems
- **Documented**: Include comments for complex code
- **Error handling**: Show proper exception handling
- **Modern**: Use object-oriented API by default

### Section Organization

1. **Introduction**: What this is about
2. **Prerequisites**: What's needed
3. **Main content**: The actual information
4. **Examples**: Working code
5. **See also**: Related content

## Continuous Documentation

### Auto-rebuild During Development

Use sphinx-autobuild for live reloading:

```bash
pip install sphinx-autobuild
sphinx-autobuild sphinx sphinx/_build/html
```

Visit http://localhost:8000 - documentation rebuilds automatically when files change.

### Link Checking

Check for broken links:

```bash
make docs-linkcheck
```

## Publishing Documentation

### GitHub Pages

1. Build documentation: `make docs`
2. Copy `sphinx/_build/html` to `docs/` directory
3. Commit and push
4. Enable GitHub Pages in repository settings

### Read the Docs

1. Connect repository to Read the Docs
2. Configure to use `sphinx/` directory
3. Documentation builds automatically on push

## Troubleshooting

### Build Errors

**"module not found" errors:**
- Ensure coremusic is built: `make build`
- Add `src/` to Python path in `conf.py`

**Extension errors:**
- Install required extensions: `pip install sphinx-rtd-theme`
- Check `extensions` list in `conf.py`

**Autodoc errors:**
- Verify module imports work
- Check function/class names are correct
- Ensure docstrings are properly formatted

### Formatting Issues

**Code blocks not formatting:**
- Check indentation (3 spaces after `.. code-block::`)
- Verify language is specified: `.. code-block:: python`

**Links broken:**
- Use correct syntax: `:doc:\`page\`` not `:doc:\`page.rst\``
- Check file paths are relative to current file

## Contributing to Documentation

We welcome documentation contributions! To contribute:

1. Fork the repository
2. Create a documentation branch
3. Make your changes following these guidelines
4. Build and verify: `make docs`
5. Submit a pull request

Good documentation contributions:
- Fix typos or unclear explanations
- Add missing examples
- Improve code samples
- Add new tutorials or recipes
- Enhance API documentation

## See Also

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Read the Docs](https://readthedocs.org/)
