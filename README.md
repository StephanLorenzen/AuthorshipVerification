# Writing style analysis in Danish High schools
This repository contains code for writing style analysis. Three related problems are considered:

* Computing writing style similarity between two texts (`sim` module).
* Detecting ghostwriters by detecting sudden changes in writing style (`av` module).
* Detecting general tendencies in writing style development and predicting writing style development (`ws` module)

The application is build around the `sim` module, which is based on deep learning methods for computing writing style similarity. The network architectures included are optimized for Danish essays from Danish high schools, but will work (with varying succes) in English texts.

In order to use the `av` and `ws` modules, the `sim`-network must first be trained, see below.

## The sim module
The `sim` module implements neural networks for computing text similarity. Training/testing method and networks are described in [1].

## The av module
The `av` module computes the similarities between an unknown text and the known texts of a claimed author, which are then combined in order to answer, whether the claimed author is the author of the unknown text. Training/testing method and networks are described in [1].

## The ws module
The `ws` module computes the similarity between a text and previous texts by the same author, in order to detect changes in writing style.

More to come...

## Publications.
[1] Stavngaard et al., Detecting Ghostwriters in High School, ESANN'19 (Submitted).
