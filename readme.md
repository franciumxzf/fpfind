## Overview

Most of the functions are implemented in the library directory `lib/`, which are then imported by the glue scripts and/or other entrypoints to test with actual data.

Data is located in the subdirectory `data/`, which can be created as a symbolic link to the actual directory so as to not import these large files into the git repository. The data repository is excluded from the git repository using `.gitignore`.

In Linux, symbolic links can be created via:

```
ln -s [FULLPATH_TO_FILE_OR_DIR] [LOCATION_OF_LINK]
```

In Windows, symbolic links are created on command prompt `cmd.exe`. Directory soft links should be specified with `/D`:

```
mklink [LOCATION_OF_LINK] [FULLPATH_TO_FILE]
mklink /D [LOCATION_OF_LINK] [FULLPATH_TO_DIR]
```

## Running

The main entry-point is `main.py`, which currently only runs `pfind` for dataset27. This is where the main testing occurs.

The test infrastructure is installed and relies on `pytest`. Tests can be further customized per our requirements, see [pytest documentation](https://docs.pytest.org/en/7.2.x/how-to/index.html). To run the tests, install the required libraries first before running pytest in the "code" subdirectory:

```
cd ${TOPDIR}
pip install -r requirements.txt

cd ${TOPDIR}/code
pytest
```

To have the py2c test work, compile the example C program in the examples directory:

```
cd ${TOPDIR}/code/example
gcc example_hello.c -o example_hello

cd ${TOPDIR}/code
pytest
```

Sample output to see from pytest, if all the tests succeed:

```
justin@coldsake:~/Documents/_local/GitHub/Clock-sync/code$ pytest
========================================== test session starts ==========================================
platform linux -- Python 3.10.6, pytest-7.2.0, pluggy-1.0.0
rootdir: /home/justin/Documents/_local/GitHub/Clock-sync/code
plugins: dash-2.7.1, anyio-3.6.2
collected 2 items                                                                                       

test/test_pfind.py .                                                                              [ 50%]
test/test_py2c_example_hello.py .                                                                 [100%]

========================================== 2 passed in 11.51s ===========================================
```

