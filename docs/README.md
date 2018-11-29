# MDSA (MOOC Data Science Analytics) Pakcage

Why MLDSA?: capable of doing any data-analytics tasks at different level for different users

MLDSA consists of 7 sub-packages:
- mldsa.core: the interface and the logic behind
- mldsa.preprocess: preprocessing tracking logs, big-query tables, and MLC tables, also encapsulation of courses and big-query tables
- mldsa.data: encapsulation of the data object: aggregation
- mldsa.nn: encapsulation of nn Modules, Layers, and Funtionals as added to / inherited from pytorch's interface
- mldsa.model: encapsulation of models
- mldsa.train: training the models on aggregations, encapsulation of train results and model comparison results
- mldsa.visualize: utility package for visualization

The expected usage of MLDSA shifts along with development process:
- Current: modeling (NN training) package based on MLC (intend to be MLM)
- Future: backend package with utility functions to be called when doing data analytics (within jupyter)
  - the way researchers use it, e.g.use the parsers in preprocessing sub-package to parse the big-query tables (and analyze the "study patterns")
- provide a distributed task management system and a resource database for data analytics at scale
- the original call-and-use way is preserved and compatible with the current pipeline
- no need to write any for-loop and track the results manually, free your hand for research
- provide clear organization and detailed views of results

## Example uses

- For 6 courses, preprocess for all interested aggregations within 6 hours (with alfad8 of 12 CPUs and non-SSD storage)

- For 3 different data-model configurations, scaling up to train ~1K models within a day (with slurm cluster of 10 GPUs)

## Documentation

Build the documentation using `Sphinx`
