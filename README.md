# Writing style analysis in Danish High schools
This repository contains code for writing style analysis. Three related problems are considered:

* Computing writing style similarity between two texts (`sim` module).
* Detecting ghostwriters by detecting sudden changes in writing style (`av` module).
* Detecting general tendencies in writing style development and predicting writing style development (`ws` module)

The application is build around the `sim` module, which is based on deep learning methods for computing writing style similarity. The network architectures included are optimized for Danish essays from Danish high schools, but will work (with varying succes) in English texts.

In order to use the `av` and `ws` modules, the `sim`-network must first be trained, see below.

## Input data format
The modules require raw data to be stored in .csv format, using semi-colon for separators. The file must contain three columns, and no header. Each line (corresponding to one text) must have the following format:

```
<authorId>;<date>;<text>
```

where `<authorId>` is a string identifying the author, `<date>` is the date the text was written/handed-in (in format: DD-MM-YYYY) and `<text>` is the text. Note, that text should be sanitized by substituting ';' with '$SC$' and newline characters with '$NL$'.

Before running any experiments, the data must be preprocessed using the `sim` module, see below.

## Configuration
Each module requires a configuration file, which might be specified during execution or located in the current directory. Each module will load the configuration file with the following priority:

1. File specified during execution
2. config-{sim,av,ws}.ini in the current directory.
3. config.ini in the current directory.

For each module, the configuration file must contain the following section:

```
[Path]
data    = <directory for storing processed data>
storage = <directory for storing models>
```

Paths must be given with a trailing '/'. Note, that `data` and `storage` can be the same for all modules.

Furthermore, each configuration file must contain a `[Default]` section, providing default parameters; see the sections below for parameters required for each module.

Note, that the same configuration file can be used for all modules. The file `config.ini.template` provides an example.

## The sim module
The `sim` module implements neural networks for computing text similarity. Training/testing method and networks are described in [1].

### Configuration
The configuration file must specify the following parameters in the `[Default]` section:

* `dataset`: which dataset to use.


### Data preprocessing


### Usage

```
usage: sim [-h] {train,test,eval} ...

Computing writing style similarity using deep learning.

positional arguments:
  {train,test,eval}

optional arguments:
  -h, --help         show this help message and exit
```

### Example

## The av module
The `av` module computes the similarities between an unknown text and the known texts of a claimed author, which are then combined in order to answer, whether the claimed author is the author of the unknown text. Training/testing method and networks are described in [1].

### Configuration
The `av` module requires a configuration file, which specifies locations for model and data storage, as well as default parameters.

### Usage
### Example

## The ws module
The `ws` module computes the similarity between a text and previous texts by the same author, in order to detect changes in writing style. The module utilizes the `sim`-module for computing similarities, while the clustering approach is described in [2].

### Configuration
The `ws` module requires a configuration file, which specifies locations for model and data storage, as well as default parameters.

### Usage
### Example

## Publications.
[1] M. Stavngaard, A. Sørensen, S. Lorenzen, N. Hjuler, S. Alstrup: _Detecting Ghostwriters in High Schools_, ESANN'19. ([arXiv](https://arxiv.org/abs/1906.01635)).
[2] S. Lorenzen, N. Hjuler, S. Alstrup: _Investigating Writing Style Development in High School_, EDM'19. ([arXiv](https://arxiv.org/abs/1906.03072)).
