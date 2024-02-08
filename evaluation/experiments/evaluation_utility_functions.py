"""
A script with utility functions to load StereoSet evaluation scores from files
to plot for visualisation.

:author: Karina Hensel
"""

import json, os
import numpy as np


def load_overall_scores(filename):
    """
    Load the overall lms, ss, iCAT scores of a number of models

    :param filename: path to file with evaluation scores
    :type: str
    :returns: model - score mapping
    :rtype: Dict[str, list)
    """

    with open(filename, "r") as f:
        data = json.load(f)

    all_scores = {}
    for i in data.keys():
        all_scores[i] = [data[i]["intrasentence"]["overall"]["LM Score"], data[i]["intrasentence"]["overall"]["SS Score"],
                    data[i]["intrasentence"]["overall"]["ICAT Score"]]

    return all_scores

def average_ss_scores(dirname):
    """
    Average ss scores of stepwise updates for each batch

    :param dirname: directory with score files for all runs
    :returns: average ss scores for each update batch
    :rtype: Dict
    """

    avg_scores = {}

    for d in os.listdir(dirname):
        with open(dirname+"/"+d, "r") as f:
            data = json.load(f)

            for b in data:
                if avg_scores.get(b) != None and avg_scores[b] != 0.0:
                    old_scores = avg_scores[b]
                    new_scores = old_scores+[data[b]["intrasentence"]["overall"]["SS Score"]]
                    avg_scores[b] = new_scores
                else:
                    avg_scores[b] = [data[b]["intrasentence"]["overall"]["SS Score"]]

    for i in avg_scores.keys():
        all_scores = avg_scores[i]
        avg = np.average(all_scores)

        avg_scores[i] = avg

    return avg_scores

def std_ss_scores(dirname):
    """
    Standard deviation of ss scores of stepwise updates for each batch

    :param dirname: directory with score files for all runs
    :returns: standard deviation for each update batch
    :rtype: Dict
    """

    std_scores = {}

    for d in os.listdir(dirname):
        with open(dirname+"/"+d, "r") as f:
            data = json.load(f)

            for b in data:
                print(b)
                if std_scores.get(b) != None and std_scores[b] != 0.0:
                    old_scores = std_scores[b]
                    new_scores = old_scores+[data[b]["intrasentence"]["overall"]["SS Score"]]
                    print(b)
                    std_scores[b] = new_scores
                else:
                    std_scores[b] = [data[b]["intrasentence"]["overall"]["SS Score"]]

    for i in std_scores.keys():
        all_scores = std_scores[i]
        std = np.std(all_scores)

        std_scores[i] = std

    return std_scores

if __name__ == '__main__':
    avg_scores_file = "../results/evaluation_stepwise_debias/stats/EN/gpt2-xl-avgs.json"
    std_scores_file = "../results/evaluation_stepwise_debias/stats/EN/gpt2-xl-stds.json"

    avg_scores = average_ss_scores("../results/evaluation_stepwise_debias/stats/EN/gpt2-xl")
    std_scores = std_ss_scores("../results/evaluation_stepwise_debias/stats/EN/gpt2-xl")

    with open(avg_scores_file, "w") as f:
        json.dump(avg_scores, f, indent=2)

    with open(std_scores_file, "w") as f:
        json.dump(std_scores, f, indent=2)