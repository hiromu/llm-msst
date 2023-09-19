# Metaphor and Sarcasm Scenario Test for Large Language Models

This repository provides a script to evaluate the capacity of LLMs for metaphor and sarcasm comprehension using the MSST.

Note that the questions of the MSST are taken from [Adachi et al. (2006)](https://pubmed.ncbi.nlm.nih.gov/16715930/).

## How to run

1. Prepare a virtualenv

```sh
$ python3 -mvenv .env
```

2. Install dependency libraries

```sh
$ source .env/bin/activate
$ pip install -r requirements.txt
```

3. Run the script

```sh
$ python main.py
```

Here, `OPENAI_API_KEY` and `REPLICATE_API_TOKEN` are required to run the script.

## Citation

```
@article{yakura2023msst,
    title    = {Evaluating large language models' ability to understand metaphor and sarcasm using a screening test for {Asperger} syndrome},
    author   = {Hiromu Yakura},
    journal  = {arXiv},
    year     = {2023},
    volume   = {2309.10744},
    numpages = {6},
    doi      = {10.48550/arXiv.2309.10744},
}
```
