## Imports
import random
import csv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pprint import pprint
import numpy
import scipy.stats

test_words = [
    "good",
    "bad",
    "sad",
    "angry",
    "happy",
    "mad",
    "excited",
    "scared",
    "bored"
]

def prepare_model():
    ## Define model name and file name
    # model_name = "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF"
    # model_file = "openhermes-2.5-mistral-7b.Q4_K_M.gguf"

    ## Define model name and file name
    model_name = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
    model_file = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

    ## Define model name and file name
    # model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    # model_file = "llama-2-7b-chat.Q8_0.gguf"

    ## Use the following model_name and model_file if you have 8gb ram or less
    # model_name = "TheBloke/Mistral-7B-OpenOrca-GGUF"
    # model_file = "mistral-7b-openorca.Q4_K_M.gguf"

    ## Use the following model_name and model_file if you have 16gb ram or less
    # model_name = "TTheBloke/vicuna-13B-v1.5-16K-GGUF"
    # model_file = "vicuna-13b-v1.5-16k.Q4_K_M.gguf"

    ## Download the model
    model_path = hf_hub_download(model_name, filename=model_file)


    model_kwargs = {
      "n_ctx": 4096,
      "n_threads": 12,
      "n_gpu_layers": -1,
      "logits_all": True,
    }

    ## Instantiate model from downloaded file
    llm = Llama(model_path=model_path, **model_kwargs)

    return llm


def load_generics():
    with open('generics.txt', 'rt') as f:
        generic_sentences = f.read().splitlines()

    # Shuffle randomly for interest
    random.shuffle(generic_sentences)

    return generic_sentences


def compute_emotion_scores_for_text(llm, text):

    test_sequence = " ".join(test_words)
    test_word_tokens = llm.tokenize(test_sequence.encode("utf-8"), add_bos=False)

    prompt = f"Please choose the emotion that best represents the following text: '{text}'\n\nEmotion: good"
    llm.reset()
    prompt_tokenized = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    eval_res = llm.eval(prompt_tokenized)  # Res is a dictionary
    print(text)
    print("     Last token", llm.detokenize([llm.eval_tokens[-1]]))
    logprobs = llm.logits_to_logprobs(llm.eval_logits)

    results = {}
    for test_word, token in zip(test_words, test_word_tokens):
        logprob = logprobs[-1][token]
        actual_prob = 2.718281828459045 ** logprob

        print("    ", llm.detokenize([token]), f"{logprob:.1f}")

        # results[llm.detokenize([token])] = f"{logprob:.1f}"
        results[test_word] = logprob

    return results


def score_generics():
    llm = prepare_model()
    generic_sentences = load_generics()

    with open('results.csv', 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(['Text'] + test_words)

        for text in generic_sentences:
            compute_emotion_scores_for_text(llm, text)
            row = [text]

            scores = compute_emotion_scores_for_text(llm, text)
            for key in test_words:
                row.append(scores[key])

            writer.writerow(row)
            f.flush()


def compute_stats_from_results_file():
    with open("results.csv", "rt") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    emotion_keys = [key for key in rows[0].keys() if key != "Text"]
    values_by_emotion = {}
    means_by_emotion = {}
    stds_by_emotion = {}
    for key in emotion_keys:
        values_for_key = [float(row[key]) for row in rows[1:]]

        mean = numpy.mean(values_for_key)
        std = numpy.std(values_for_key)

        print("key average", key, f"{mean:.2f}", "std", f"{std:.2f}")

        values_by_emotion[key] = values_for_key
        means_by_emotion[key] = mean
        stds_by_emotion[key] = std

    return emotion_keys, means_by_emotion, stds_by_emotion


def main():
    llm = prepare_model()

    emotion_keys, means_by_emotion, stds_by_emotion = compute_stats_from_results_file()

    results = compute_emotion_scores_for_text(llm, "You are a true son of a bitch aren't you. I hate you!")
    for key in emotion_keys:
        z_score = (float(results[key]) - means_by_emotion[key]) / stds_by_emotion[key]
        results[key] = f"{50 + z_score * 15:.2f}"

    pprint(results)


if __name__ == "__main__":
    main()


