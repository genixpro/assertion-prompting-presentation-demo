import csv
import numpy
import matplotlib.pyplot as plt
import scipy.stats


def load_results():
    with open("results.csv", "rt") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return rows


def add_normalized_emotion_scores_to_rows(rows, means_by_emotion, stds_by_emotion, emotion_keys):
    for row in rows:
        print(row["Text"])
        for key in emotion_keys:
            z_score = (float(row[key]) - means_by_emotion[key]) / stds_by_emotion[key]
            cumulative_probability = scipy.stats.norm.cdf(z_score)

            print("    ", key, f"{z_score:.2f}", f"{cumulative_probability:.2f}")
            row[key + " normalized"] = f"{z_score:.2f}"


def output_top_bottom_ten_for_emotions(rows, emotion_keys):
    for key in emotion_keys:
        # Now we can sort the rows by the normalized value
        rows = sorted(rows, key=lambda row: float(row[key + " normalized"]), reverse=True)

        # Show the top 10 and bottom 10
        print("Top Ten for ", key)
        for row in rows[:10]:
            print("    ", row["Text"], row[key + " normalized"])

        print("Bottom Ten for ", key)
        for row in rows[-10:]:
            print("    ", row["Text"], row[key + " normalized"])


def compute_emotion_stats_for_results(rows, emotion_keys):
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

    return values_by_emotion, means_by_emotion, stds_by_emotion


def output_emotion_histograms(rows, emotion_keys):
    for key in emotion_keys:
        values_for_key = [float(row[key]) for row in rows[1:]]

        plt.hist(values_for_key, bins=10)
        plt.title(key)
        plt.show()
        plt.close()


def main():
    rows = load_results()

    emotion_keys = [key for key in rows[0].keys() if key != "Text"]

    values_by_emotion, means_by_emotion, stds_by_emotion = compute_emotion_stats_for_results(rows, emotion_keys)

    output_emotion_histograms(rows, emotion_keys)

    plt.boxplot([values_by_emotion[key] for key in emotion_keys], labels=emotion_keys)
    plt.show()
    plt.close()

    add_normalized_emotion_scores_to_rows(rows, means_by_emotion, stds_by_emotion, emotion_keys)

    output_top_bottom_ten_for_emotions(rows, emotion_keys)


if __name__ == "__main__":
    main()
