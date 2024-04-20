# PyGenOpt

A generic framework for writing linear optimization problems in Python.

**The solver's Python package must be installed for this framework to work.**

If this package was useful in any way, please let me know.

### Installation

Installing from source

```
git clone https://github.com/guifcoelho/pygenopt
pip install ./pygenopt
```

Each specific solver package must also be installed. The current supported solvers are:

- HiGHS (default, installed with PyGenOpt):

```
pip install highspy==1.7.1.dev1
```

- Xpress:

```
pip install xpress
```
