import datasets
import os
import pandas as pd
import random
import yaml
import string
import nltk
import math
import re
import collections


nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# === FILE LOADING =================================================================================
def read_qna_file(qna_path):
    """Safe-load a qna yaml file into a dict"""
    with open(qna_path) as stream:
        return yaml.safe_load(stream)


def recursive_qna_search(directory):
    """Parse all files named "qna.yaml" in a directory + its subdirectories.

    Returns: A list of pairs [path, parsed yaml
    """
    paths = os.listdir(directory)
    qnas = []
    for p in paths:
        subpath = os.path.join(directory, p)
        if os.path.isdir(subpath):
            qnas += recursive_qna_search(subpath)
        elif p == "qna.yaml":
            qnas.append([subpath, read_qna_file(subpath)])
    return qnas


def load_taxonomy(parent_dir):
    """Given a parent taxonomy directory, load all qna files within."""
    dirs = ["foundational_skills", "compositional_skills", "knowledge"]
    dirs = [os.path.join(parent_dir, d) for d in dirs]

    qna_yamls = []
    for dir_ in dirs:
        qna_yamls += recursive_qna_search(dir_)

    return qna_yamls


# === TAXONOMY PREPROCESSING =======================================================================
def text_standardization(text):
    """Cleaning function to apply to extracted qna text fields (e.g., answer, question, context)"""
    if text is not None:
        return text.strip()
    else:
        return text


def preprocess_qnas(qna_yamls):
    """Standardize the qnas and extract relevant fields, return as Dataframe"""
    preprocessed_qnas = []
    for filename, qna in qna_yamls:
        for example in qna['seed_examples']:
            if "task_description" in qna:
                description = text_standardization(qna['task_description'])
            elif "document_outline" in qna:
                description = text_standardization(qna["document_outline"])
            else:
                description = None

            if "questions_and_answers" in example:
                for subexample in example['questions_and_answers']:
                    preprocessed_qnas.append({
                        "question": text_standardization(subexample['question']),
                        "answer": text_standardization(subexample['answer']),
                        "context": text_standardization(example['context']),
                        "source": filename,
                        "description": description
                    })
            else:
                preprocessed_qnas.append({
                    "question": text_standardization(example['question']),
                    "answer": text_standardization(example['answer']),
                    "context": text_standardization(example.get('context')),
                    "source": filename,
                    "description": description
                })

    return pd.DataFrame(preprocessed_qnas)


def strip_common_filename_path(filename_column):
    """strip out any universal full path references in the source taxonomy column
    e.g., turn /etc/abc/taxonomy/foundational_skills/logic -> foundational_skills/logic"""
    common_path = os.path.commonpath(filename_column.tolist())
    return filename_column.apply(lambda p: os.path.relpath(p, common_path))


# === QUESTION GENERATION ==========================================================================
def text_splitter(text):
    words = []
    for word in text.lower().split():
        proc_word = word.strip(string.punctuation).strip()
        if proc_word:
            words.append(proc_word)
    return words


def offset_tokenize_and_parts_of_speech(text):
    tail = text
    accum = 0
    tokens = parts_of_speech(text.strip())
    info_tokens = []

    relevant_tags = [
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNS",
    "NNP",
    "NNPS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB"]

    for tok, p in tokens:
        if p not in relevant_tags:
            continue
        scaped_tok = re.escape(tok)
        m = re.search(scaped_tok, tail)
        try:
            start, end = m.span()
        except Exception as e:
            raise e
        # global offsets
        gs = accum + start
        ge = accum + end
        accum += end
        # keep searching in the rest
        tail = tail[end:]
        info_tokens.append((tok, p, (gs, ge)))
    return info_tokens


def parts_of_speech(text):
    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)


def substitute_parts_of_speech(original_text, text_words, replacement_word_set, target_parts_of_speech):
    replace_pairs = []
    intra_question_swap = []
    for target_part_of_speech in target_parts_of_speech:
        correct_answer_nouns = [(w, part_of_speech, (ts, te)) for w, part_of_speech, (ts, te) in text_words if part_of_speech == target_part_of_speech]

        # choose some random nouns to replace
        num_to_swap = min(3,  math.ceil(len(correct_answer_nouns) * .33))
        to_replace = random.sample(correct_answer_nouns, num_to_swap)

        # grab other examples of target part of speech from the rest of the document
        other_words = [word for word, part_of_speech in replacement_word_set if part_of_speech == target_part_of_speech]
        if len(correct_answer_nouns) > 5:
            intra_question_swap.append(target_part_of_speech)
            other_words = [w for (w, _, _) in correct_answer_nouns]

        # pair each noun-to-replace with a replacement
        if len(to_replace) == 0:
            continue

        if len(other_words) == 1 and to_replace[0][0] == other_words[0]:
            continue

        for w in to_replace:
            replacement = w[0]
            while w[0] == replacement:
                replacement = random.choice(other_words)
            replace_pairs.append((w, replacement))

    # sort the replacement pairs by their order of occurence in the original answer
    replace_pairs = sorted(replace_pairs, key=lambda x: x[0][2][0])

    # build a new answer with the replaced nouns in place of the originals
    answer_replaced = ""
    for idx, ((word, _, (start, end)), replacement) in enumerate(replace_pairs):
        if idx == 0:
            answer_replaced += original_text[:start]

        if word[0].isupper():
            answer_replaced += replacement.title()
        else:
            answer_replaced += replacement.lower()

        if idx != len(replace_pairs) - 1:
            answer_replaced += original_text[end:replace_pairs[idx + 1][0][2][0]]
        else:
            answer_replaced += original_text[end:]

    # if intra_question_swap:
    #     print("\n\n== Intra question swap", "="*75)
    #     print("Swapped:", intra_question_swap)
    #     print("Original: ")
    #     print(original_text)
    #
    #     print()
    #     print("New:")
    #     print(answer_replaced)

    return answer_replaced

def generate_questions(df, generation_method="same_source", num_choices=4):
    """ Generate multiple-choice questions for lm-eval from the taxonomy

    Keyword args:
    df -- The taxonomy dataframe
    generation_method -- the method by which to generate questions
    num_choices -- The number of choices to generate per question
    """

    sources = df['source'].unique()
    questions = []
    shuffler = list(range(num_choices))

    fill_answers = ["Other.", "I do not know.", "None of the above"]

    for source_idx, source in enumerate(sources):
        same_source = df['source'] == source
        same_source_df = df[same_source]
        other_source_df = df[~same_source]
        question_idx = 0

        available_replacement_words = set()
        if "substitution" in generation_method:
            for i, other_row in same_source_df.iterrows():
                available_replacement_words.update(parts_of_speech(other_row['question']))
                available_replacement_words.update(parts_of_speech(other_row['answer']))
                if other_row['context'] is not None:
                    available_replacement_words.update(parts_of_speech(other_row['context']))
            available_replacement_words = [(word.replace("_", " "), p_of_speech) for (word, p_of_speech) in available_replacement_words if not any(char in word for char in string.punctuation)]

        for i, row in same_source_df.iterrows():
            if generation_method == "same_source":
                correct_answer = row['answer']
                if (len(same_source_df) >= num_choices):
                    answers = same_source_df.drop(i).sample(num_choices - 1)['answer'].tolist()
                else:
                    answers = same_source_df.drop(i)['answer'].tolist()
                    num_to_add = num_choices - len(answers)
                    answers += fill_answers[:num_to_add]

            elif generation_method == "cross_source":
                correct_answer = row['answer']
                if (len(other_source_df) >= num_choices):
                    answers = other_source_df.sample(num_choices - 1)['answer'].tolist()
                else:
                    raise ValueError(
                        "There are not enough entries in the entire set of taxonomies to generate a full set of {} choices.")

            elif generation_method == "word_choice":
                correct_answer_words = text_splitter(row['answer'])
                n_words = min(15, len(correct_answer_words))
                correct_answer = ", ".join(random.sample(correct_answer_words, n_words))
                wrong_answers = same_source_df.drop(i).sample(num_choices - 1, replace=True)['answer'].tolist()
                answers = []
                for wrong_answer in wrong_answers:
                    wrong_answer_text = text_splitter(wrong_answer)
                    while len(wrong_answer_text) < n_words:
                        wrong_answer_text += text_splitter(other_source_df.sample(1)['answer'].iloc[0])
                    answers.append(", ".join(random.sample(wrong_answer_text, n_words)))

            elif generation_method == "part_of_speech_substitution":
                correct_answer = row['answer']
                correct_answer_words = offset_tokenize_and_parts_of_speech(correct_answer)

                num_proper_nouns = len([word for (word, p_of_speech, _) in correct_answer_words if p_of_speech in ['NNP', 'NNPS']])


                target_parts_of_speech = ['NNP', "NNPS"]
                if num_proper_nouns < 5:
                    target_parts_of_speech += ["NN", "NNS"]

                answers = []
                for _ in range(num_choices - 1):
                    answers.append(
                        substitute_parts_of_speech(
                            correct_answer, correct_answer_words, available_replacement_words, target_parts_of_speech
                        ))

            choices = [correct_answer] + answers
            random.shuffle(shuffler)
            correct_answer_pos = shuffler.index(0)
            shuffled_choices = [choices[i] for i in shuffler]

            generated_question = row
            generated_question['choices'] = shuffled_choices
            generated_question["answer_idx"] = correct_answer_pos
            generated_question["answer_text"] = correct_answer
            generated_question['answer_generation_method'] = generation_method
            generated_question['question_index'] = question_idx

            question_idx+=1
            questions.append(generated_question)
    return pd.DataFrame(questions)


def generate_prompt(question_dataframe_row):
    # if question_dataframe_row["description"]:
    #     prompt += " The following is the intent or description of the source document: {}".format(
    #         question_dataframe_row["description"]
    #     )

    use_context = question_dataframe_row["context"] and "grounded" in question_dataframe_row['source']

    if question_dataframe_row['answer_generation_method'] in ["same_source", "cross_source", "part_of_speech_substitution"]:
        prompt = "The following is a multiple choice question designed to evaluate large language models:"
        prompt += "\n\n" + question_dataframe_row['question'] + "\n\n"
        if use_context:
            prompt += "\nThe question has the following context: " + question_dataframe_row['context'] + "\n\n"
        prompt += "Please select the response that is the most appropriate answer to the question:\n\n"
    elif question_dataframe_row['answer_generation_method'] == "word_choice":
        prompt = "The following is a multiple choice question designed to evaluate large language models:"
        prompt += "\n\n" + question_dataframe_row['question'] + "\n\n"
        if use_context:
            prompt += "\nThe question has the following context: " + question_dataframe_row['context']  + "\n\n"
        prompt += "Please select the words that are most likely to appear in a correct answer to the question:\n\n"

    answer_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, c in enumerate(question_dataframe_row['choices']):
        prompt += "{}. {}\n\n".format(answer_letters[i], c)
    prompt += "\nAnswer:"
    return prompt


# === MAIN =========================================================================================
def generate_taxonomy_dataset(taxonomy_path, log_prints=False):
    print("Generating taxonomy from {}".format(taxonomy_path))

    # load qna yaml files
    qna_yamls = load_taxonomy(taxonomy_path)

    # preprocess qnas + format as dataframe
    qna_df = preprocess_qnas(qna_yamls)
    qna_df["source"] = strip_common_filename_path(qna_df["source"])

    # generate questions
    question_df = pd.concat([
        # generate_questions(qna_df, generation_method="same_source"),
        generate_questions(qna_df, generation_method="cross_source"),
        # generate_questions(qna_df, generation_method="word_choice"),
        generate_questions(qna_df, generation_method="part_of_speech_substitution")
    ])

    # add model prompts to df
    question_df["prompt"] = question_df.apply(generate_prompt, axis=1)

    if log_prints:
        for _, row in question_df.iterrows():
            print("="*75)
            print("Desired evaluation:", row["description"])
            print("Taxonomy Source:", row["source"])
            print(row["prompt"])
            print("\n\n")

    return datasets.Dataset.from_pandas(question_df)


# === DATASET DEFINITIONS ==========================================================================
class InstructlabTaxonomyEval(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")
    taxonomy_dataset = None

    def generate_dataset(self):
        if "INSTRUCTLAB_TAXONOMY_PATH" not in os.environ:
            raise ValueError("Environment variable INSTRUCTLAB_TAXONOMY_PATH must be set.")

        taxonomy_path = os.environ["INSTRUCTLAB_TAXONOMY_PATH"]
        self.taxonomy_dataset = generate_taxonomy_dataset(taxonomy_path)

    def _info(self):
        if self.taxonomy_dataset is None:
            self.generate_dataset()
        return self.taxonomy_dataset.info

    def _split_generators(self, dl_manager):
        if self.taxonomy_dataset is None:
            self.generate_dataset()
        return [datasets.SplitGenerator(name=datasets.Split.TEST,),]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self):
        if self.taxonomy_dataset is None:
            self.generate_dataset()
        return enumerate(self.taxonomy_dataset)


if __name__ == '__main__':
    taxonomy_path = os.environ["INSTRUCTLAB_TAXONOMY_PATH"]
    taxonomy_dataset = generate_taxonomy_dataset(taxonomy_path, log_prints=False)

    print(len(taxonomy_dataset))
