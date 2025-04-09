import collections

import datasets
import os
import pandas as pd
import pathlib


# === CONSTANTS ====================================================================================
class Operation:
    def __init__(self, name, plaintext, function):
        self.name = name
        self.plaintext = plaintext
        self.function = function

    def __call__(self, x, y):
        return self.function(x, y)


ADDITION = Operation("addition", "plus", lambda x, y: x + y)
SUBTRACTION = Operation("subtraction", "minus", lambda x, y: x - y)
MULTIPLICATION = Operation("multiplication", "times", lambda x, y: x * y)
DIVISION = Operation("division", "divided by", lambda x, y: x // y)
MODULO = Operation("modulo", "modulo", lambda x, y: x % y)
EXPONENTIATION = Operation("exponentiation", "to the power of", lambda x, y: x ** y)

OPERATIONS = [
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    MODULO,
    EXPONENTIATION
]

ANSWER_LETTERS = "ABCD"
null_dataset_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "data_template.csv")

# === GENERATE DATASET =============================================================================
class LinearCongruentialGenerator():
    """Guarantee that the random numbers are consistent regardless of Python version or OS"""

    # all possible 4-item shuffles
    shuffle_combinations = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1),
                            (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
                            (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0),
                            (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)]

    def __init__(self, seed=1):
        self.state = seed

    def random(self, lower_bound, upper_bound):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return (self.state % (upper_bound - lower_bound)) + lower_bound

    def four_item_shuffle(self, original_list):
        """randomly shuffle a list of four items"""
        return [original_list[idx] for idx in self.shuffle_combinations[self.random(0, 24)]]

    def random_list(self, lower_bound, upper_bound, size, forbidden_numbers=None):
        """Return a random list of size n, where every value is unique and not in forbidden number list"""
        if size > (upper_bound-lower_bound):
            raise ValueError(f"Size {size} is greater than total available numbers ({upper_bound-lower_bound}) between upper and lower bound.")

        rands = set()
        while len(rands) < size:
            rand = self.random(lower_bound, upper_bound)
            if forbidden_numbers and rand not in forbidden_numbers:
                rands.add(rand)
        return list(rands)


def operation_value_getter(operation_name, lcg, constraints=None):
    """Get values of x,y that are appropriate for a given operation"""
    if constraints: # keep generated values close to originals
        x, y = constraints
        if operation_name == EXPONENTIATION.name:  # ensure that answers are not overly large
            return lcg.random(max(x-5, 0), x+5), lcg.random(max(y-5, 0), y+5)
        elif operation_name == DIVISION.name:  # ensure that all answers are integers
            x = lcg.random(max(x-5, 1), x+5)
            y = lcg.random(max(y-5, 1), y+5)
            return x * y, x
        elif operation_name == MODULO.name:  # prevent divide by zero errors
            return lcg.random(max(x-3, 0), x+3), lcg.random(max(y-3, 1), x+3)
        else:
            return lcg.random(x-3, x+3), lcg.random(y-3, y+3)

    else: # just get new good values
        if operation_name == EXPONENTIATION.name:  # ensure that answers are not overly large
            return lcg.random(0, 21), lcg.random(0, 6)
        elif operation_name == DIVISION.name:  # ensure that all answers are integers
            x = lcg.random(1, 100)
            y = lcg.random(1, 100)
            return x * y, x
        elif operation_name == MODULO.name:  # prevent divide by zero errors
            x = lcg.random(0, 100)
            y = lcg.random(1, x+3)
            return x, y
        else:
            return lcg.random(0, 100), lcg.random(0, 100)


def generate_questions(operation_list, n_questions=5000, random_seed=0):
    """Generate a random set of arithmetic questions"""
    lcg = LinearCongruentialGenerator(random_seed)
    question_format = "What is {} {} {}?"
    questions = []

    for operation in operation_list:
        for q in range(n_questions):
            # pick an operation and two integers at random, compute answer
            x, y = operation_value_getter(operation.name, lcg)
            answer = operation(x, y)

            # generate 3 plausible wrong answers
            possible_answers = {answer}
            while len(possible_answers) < 4:
                new_candidate = operation_value_getter(operation.name, lcg, constraints=[x, y])
                possible_answers.add(operation(*new_candidate))
            possible_answers = lcg.four_item_shuffle(list(possible_answers))

            # format question into a full prompt
            prompt = question_format.format(x, operation.plaintext, y) + "\n\n"
            prompt += "\n\nPlease select the correct answer:\n"

            # format the answer choices
            for i, a in enumerate(possible_answers):
                prompt += "{}. {}\n".format(ANSWER_LETTERS[i], a)
            prompt += "\nAnswer:"

            questions.append({
                "prompt": prompt,
                "answer_index": possible_answers.index(answer),
            })
    return pd.DataFrame(questions)


def print_questions_df(questions_df):
    """Pretty-print a set of questions"""
    for _, question in questions_df.iterrows():
        print("=" * 84)
        print(question["prompt"])
        answer_letter = ANSWER_LETTERS[question["answer_index"]]
        print(f"\nCorrect Answer: {answer_letter}")
        print("\n")


# === DATASET DEFINITIONS ==========================================================================
def create_dataset(operation_list, random_seed=0):
    """Overwrite the blank placeholder dataset with one we generate"""
    return datasets.Dataset.from_pandas(
        generate_questions(operation_list=operation_list, random_seed=random_seed, n_questions=5), split=datasets.Split.TRAIN
    )


# === SUB-DATASET DEFINITIONS =======================================================================
def create_addition_dataset(placeholder_dataset: datasets.Dataset):
    """Overwrite the blank placeholder dataset with one we generate"""
    return create_dataset([ADDITION], random_seed=1)


def create_subtraction_dataset(placeholder_dataset: datasets.Dataset):
    """Overwrite the blank placeholder dataset with one we generate"""
    return create_dataset([SUBTRACTION], random_seed=2)


def create_multiplication_dataset(placeholder_dataset: datasets.Dataset):
    """Overwrite the blank placeholder dataset with one we generate"""
    return create_dataset([MULTIPLICATION], random_seed=3)


def create_division_dataset(placeholder_dataset: datasets.Dataset):
    """Overwrite the blank placeholder dataset with one we generate"""
    return create_dataset([DIVISION], random_seed=4)


def create_modulo_dataset(placeholder_dataset: datasets.Dataset):
    """Overwrite the blank placeholder dataset with one we generate"""
    return create_dataset([MODULO], random_seed=5)


def create_exponentiation_dataset(placeholder_dataset: datasets.Dataset):
    """Overwrite the blank placeholder dataset with one we generate"""
    return create_dataset([EXPONENTIATION], random_seed=6)


if __name__ == "__main__":
    df = generate_questions(OPERATIONS, 5 ,2 )
    print_questions_df(df)

