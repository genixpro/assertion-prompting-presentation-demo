# Imports
import random
import csv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pprint import pprint
import numpy

emotion_words = [
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


def load_llm_model():
    # Define model name and file name
    model_name = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
    model_file = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

    # Download the model
    model_path = hf_hub_download(model_name, filename=model_file)

    model_kwargs = {
      "n_ctx": 4096,
      "n_threads": 12,
      "n_gpu_layers": -1,
      "logits_all": True,
    }

    # Instantiate model from downloaded file
    llm = Llama(model_path=model_path, **model_kwargs)

    return llm


def load_generic_sentence_dataset():
    with open('generics.txt', 'rt') as f:
        generic_sentences = f.read().splitlines()

    # Shuffle randomly for interest
    random.shuffle(generic_sentences)

    return generic_sentences


def compute_emotion_scores_for_text(llm, text):
    # Figure out the correct token for each of the emotion words
    emotion_word_sequence = " ".join(emotion_words)
    emotion_word_tokens = llm.tokenize(emotion_word_sequence.encode("utf-8"), add_bos=False)

    # Create a prompt that gets the model to predict an emotion word as its final token
    prompt = f"Please choose the emotion that best represents the following text: '{text}'\n\nEmotion: good"
    llm.reset()
    prompt_tokenized = llm.tokenize(prompt.encode("utf-8"), add_bos=True)

    # Process the prompt through the language model
    llm.eval(prompt_tokenized)

    # Fetch the log probabilities from the model
    log_probabilities = llm.logits_to_logprobs(llm.eval_logits)

    # For each emotional
    results = {}
    for emotion_word, emotion_word_token in zip(emotion_words, emotion_word_tokens):
        log_probability = log_probabilities[-1][emotion_word_token]

        print("    ", llm.detokenize([emotion_word_token]), f"{log_probability:.1f}")

        results[emotion_word] = log_probability

    return results


def score_generics():
    llm = load_llm_model()
    generic_sentences = load_generic_sentence_dataset()

    with open('results.csv', 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(['Text'] + emotion_words)

        for text in generic_sentences:
            compute_emotion_scores_for_text(llm, text)
            row = [text]

            scores = compute_emotion_scores_for_text(llm, text)
            for key in emotion_words:
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
    llm = load_llm_model()

    # score_generics()

    emotion_keys, means_by_emotion, stds_by_emotion = compute_stats_from_results_file()

    results = compute_emotion_scores_for_text(llm, "You are a true son of a bitch aren't you. I hate you!")
    for key in emotion_keys:
        z_score = (float(results[key]) - means_by_emotion[key]) / stds_by_emotion[key]
        results[key] = f"{z_score:.2f}"

    pprint(results)


if __name__ == "__main__":
    main()


