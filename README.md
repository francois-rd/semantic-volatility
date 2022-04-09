# The Semantic Volatility of Neologisms

This is the code repository for my CSC2611 course project at the University of Toronto.

**Semantic volatility** is idea that new words (i.e., neologisms) may undergo significantly more semantic change in the early days of their introduction to a language compared to already existing words. This project aims to measure that behaviour.

## Installation

Install the required libraries listed in `requirements.txt`.

Install the following external resources:
+ The `words` corpus from NLTK: http://www.nltk.org/nltk_data/
  ~~~
  >>> import nltk
  >>> nltk.download()
  ~~~
+ The `en_core_web_sm` model from SpaCy: https://spacy.io/models/en
  ~~~
  python -m spacy download en_core_web_sm
  ~~~

Add `semantic-volatility/src` to `PYTHONPATH`.

## Replicating the main results

This project runs *way too slowly* to be done all in one session. Instead, individual steps are executed as subcommands of the `src/main.py` program. Each step is independently configurable using a JSON-style config file. Taken together, the steps form a pipeline from data download all the way to final result visualization. Running:

~~~
python src/main.py -h
~~~

produces a list of each pipeline step (in order) as well as a brief help message. Some steps rely on the output of one or more of the previous steps. To execute a step, run:

~~~
python src/main.py <step_name> --config experiments/main/pipeline/<step_name>.json
~~~

where `<step_name>` is one of the valid steps. Some steps take a very long time to run. In particular, `download`, `preprocess`, and `bert` all take *days or weeks* to run. Therefore, to simply replicate the figures and tables from the final report, I advise running only the last two steps of the pipeline:

~~~
python src/main.py plot-ts --config experiments/main/pipeline/plot-ts.json
python src/main.py plot-stats --config experiments/main/pipeline/plot-stats.json 
~~~

These two steps only rely on the output of the step `time-series` and not on any previous steps. The output of `time-series` is provided in `experiments/main/model/time_series`.

Each of these steps only takes a few seconds to run and produces output files in `experiments/main/results`, some of which were used in the final report. Feel free to explore the plots that didn't make the cut to gain a richer understanding of the main results.

## Additional notes

+ Contact me directly to access any of the other intermediate output files.
+ Running `python src/ppt.py` produces the plots for the PowerPoint presentation.
+ All other details are in the final report.
