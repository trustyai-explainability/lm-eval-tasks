# Tiny Offline Arithmetic
Tiny Offline Arithmetic is a task designed to run without requiring the download of any data or 
any remote code execution, useful for testing the lm-evaluation-harness in sandboxed or disconnected 
environments. To do this, the task generates the entirety of its question corpus at runtime, using
a [linear congruential generator](https://en.wikipedia.org/wiki/Linear_congruential_generator) and a
fixed set of seeds to ensure deterministic question generation regardless of Python environment. 

The evaluation itself consists of simple arithmetic questions involving one of six operations:
addition, subtraction, multiplication, division, modulos, or exponentiation. Two integers are chosen at random,
and the result of the operation over those integers is the target answer of the question. The correct 
answer is presented alongside three plausible incorrect answers in a multiple choice format, e.g.:

```
What is 47 minus 8?

A. -45
B. -37
C. 40
D. 39
```

### Paper

N/A

### Citation

```
@misc{Geada, title={Tiny Offline Arithmetic}, url={https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks}, journal={Github}, author={Geada, Rob}} 
```

### Groups, Tags, and Tasks

#### Groups

* `tiny_offline_arithmetic`: A tiny dataset that requires no online access or remote code execution, for 

#### Tags

N/A

#### Tasks
* `tiny_offline_arithmetic_addition`: Multiple choice arithmetic questions involving addition of two integers, in the form `x + y`
* `tiny_offline_arithmetic_subtraction`: Multiple choice arithmetic questions involving subtraction of two integers, in the form `x - y`
* `tiny_offline_arithmetic_multiplication`: Multiple choice arithmetic questions involving multiplication of two integers, in the form `x * y`
* `tiny_offline_arithmetic_division`: Multiple choice arithmetic questions involving division of two integers, in the form `x / y`, where y is guaranteed to be an integer divisor of x.
* `tiny_offline_arithmetic_modulo`: Multiple choice arithmetic questions involving modulos of two integers, in the form `x % y`
* `tiny_offline_arithmetic_exponentiation`: Multiple choice arithmetic questions involving exponentiation of two integers, in the form `x ** y`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature? 
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
