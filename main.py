import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
import re
import logging
import sys
import multiprocessing

TEST_SET_PERCENT = 0.3
BUDGET = 10**4
DATA_FILE = 'password_tips_english.csv'
EOL_TAG = '_eol_'
BOL_TAG = '_bol_'


class PasswordEstimator(BaseEstimator):
    """This find the word with the highest probability to be a password within the list of words

    Args:
        word_dicts_before: this is the dictionary of words found before the password ordered by 
            their distance, for example:
        [
            // words found at distand d = 1 from the password
            {
                ':': 1,
                'password': 0.4,
                'coffee': 0.1,
                'saxophone': 0.000001
            },
            // words found at distand d = 2 from the password
            {
                ':': 0.3,
                'password': 1,
                'wifi': 0.7
                'saxophone': 0.0000001
            }
        ]
        word_dicts_after: this is the dictionary of words found after the password ordered by 
            their distance
        before_cutoff: number of words to take into account before the passwords
        after_cutoff: number of words to take into account after the passwords
        before_exponential_factor: the exponentional factor that will give the ponderation of
            words before the password
        after_exponential_factor: the exponentional factor that will give the ponderation of
            words after the password
        after_factor: the factor to which words after the password account in comparison with
            words before the password
        min_password_length: the number of character minimum for a word to be considered as a
            password
        margin_cutoff: we can look not only at the frequency of the word at that distance of the
            password but also see the match we get if we move the word to the left or right
        margin_factor: the ponderation of the match on the margin of the word found
        eol_factor: the ponderation to give to newline
    """

    def __init__(
            self, before_cutoff=10, after_cutoff=10, before_exponential_factor=1,
            after_exponential_factor=1, after_factor=0.5, min_password_length=4, margin_cutoff=10,
            margin_factor=0.05, margin_exponential_factor=1, eol_factor=0.6, bol_factor=0.05):
        self.before_cutoff = before_cutoff
        self.after_cutoff = after_cutoff
        self.before_exponential_factor = before_exponential_factor
        self.after_exponential_factor = after_exponential_factor
        self.after_factor = after_factor
        self.min_password_length = min_password_length
        self.margin_cutoff = margin_cutoff
        self.margin_factor = margin_factor
        self.margin_exponential_factor = margin_exponential_factor
        self.eol_factor = eol_factor
        self.bol_factor = bol_factor

    def add_words(self, words, word_dicts, max_words):
        """Add words found in the training set to the word dictionary

        Args:
            words: the list of words to add order by their distance to the password
            word_dicts: the relevant word dictionaries to add to
            max_words: the number of words we want to add

        Returns:
            The filled word dictionaries
        """
        for i, word in enumerate(words):
            if i >= max_words:
                break

            if word not in word_dicts[i]:
                word_dicts[i][word] = 0
            word_dicts[i][word] += 1

        return word_dicts

    def normalize(self, word_dicts):
        """Normalize the words found

        Args:
            word_dicts: the word dictionaries to be normalized

        Returns:
            The normalized word dictionaries
        """
        for i, word_dict in enumerate(word_dicts):
            max_val = max(word_dict.values())
            word_dicts[i] = {k: v / max_val for k, v in word_dict.items()}
        return word_dicts

    def get_tip_before_after_password(self, tip, password):
        """Split the tip in the string before and after the password

        Args:
            tip: the tip containing the password
            password: the known password in the tip

        Returns:
            A tupble containing the string before and the string after the password
        """

        index_password = tip.index(password)
        return tip[:index_password], tip[index_password + len(password):]

    def get_words(self, tip, reverse=False, bol=False, eol=False):
        """Split the string into words base on a regex

        Args:
            tip: the tip to be split
            reverse: reverse the words if necessary
            bol: contains beginning of line
            eol: contains end of line

        Returns:
            The list of words
        """
        # We consider the ponctuation as words on their own
        # Quotes are ignored
        words_case_sensitive = []
        if bol:
            words_case_sensitive += [BOL_TAG]
        words_case_sensitive += re.findall(r"[\w\/\\#$%@^_+*<=>&-]+|[.,!?;:]", tip)
        if eol:
            words_case_sensitive += [EOL_TAG]

        words_case_insensitive = [s.lower() for s in words_case_sensitive]

        if reverse:
            return list(reversed(words_case_sensitive)), list(reversed(words_case_insensitive))

        return words_case_sensitive, words_case_insensitive

    def fit(self, X, y):
        """Train the model

        Args:
            X: the training set containing the tips
            y: the list of passwords matching the tips
        """
        self.word_dicts_before = [{} for _ in range(0, self.before_cutoff)]
        self.word_dicts_after = [{} for _ in range(0, self.after_cutoff)]

        for i, tip in X.iteritems():
            try:
                tip_before, tip_after = self.get_tip_before_after_password(tip, y[i])
            except ValueError:
                continue

            _, words_before = self.get_words(tip_before, reverse=True, bol=True)
            _, words_after = self.get_words(tip_after, eol=True)
            self.word_dicts_before = self.add_words(
                words_before, self.word_dicts_before, self.before_cutoff)
            self.word_dicts_after = self.add_words(
                words_after, self.word_dicts_after, self.after_cutoff)

        self.word_dicts_before = self.normalize(self.word_dicts_before)
        self.word_dicts_after = self.normalize(self.word_dicts_after)
        return self

    def ponderate(self, exponential_factor, distance):
        """Ponderate the word score using an inverse exponentional

        Args:
            exponential_factor: the exponentional factor
            distance: the x axis

        Returns:
            The ponderator
        """
        return np.exp(-1 * exponential_factor * distance)

    def get_score_margin(self, left_border, right_border, word, index, word_dicts):
        """Calculate the score given by the left or right margin of the word

        Args:
            left_border: the left side of the interval we are looking at
            right_border: the right side of the interval we are looking at
            word: the word we are calculating the score for
            index: the distance from the password the word is at
            word_dicts: the dictionary of word counts

        Returns:
            The score given by this margin
        """
        score = 0
        for j in range(left_border, right_border):
            if word in word_dicts[j]:
                distance = np.absolute(j - index) - 1
                score += word_dicts[j][word] * self.ponderate(
                    self.margin_exponential_factor, distance)
        return score

    def get_score(self, words, word_dicts, max_words, exponential_factor):
        """Calculate the likeliness that a potential word is a password based on the list of words
            after or before the password

        Args:
            words: the list of words to consider
            word_dicts: the relevant word dictionary
            max_words: the number of words to consider
            exponential_factor: the exponentional factor for the word ponderation based on
                their distance to the password

        Returns:
            The score of this side of the password
        """
        score = 0
        for i in range(0, max_words):
            word = None
            word_dict = word_dicts[i]
            if i < len(words):
                word = words[i]

            if word not in word_dict:
                continue

            word_score = word_dict[word]

            # margin right
            left_border = i + 1
            right_border = min(i + 1 + self.margin_cutoff, max_words)
            score_right_margin = self.get_score_margin(left_border, right_border, word, i, word_dicts)
            word_score += score_right_margin * self.margin_factor

            # margin left
            left_border = max(i - 1 - self.margin_cutoff, 0)
            right_border = i - 1
            score_left_margin = self.get_score_margin(left_border, right_border, word, i, word_dicts)
            word_score += score_left_margin * self.margin_factor

            word_score *= self.ponderate(exponential_factor, i)
            if word == EOL_TAG:
                word_score *= self.eol_factor
            if word == BOL_TAG:
                word_score *= self.bol_factor

            score += word_score
        return score

    def predict(self, X):
        """Predict a list of password for the given tips

        Args:
            X: the list of tips

        Returns:
            The list of predicted passwords
        """

        y = []
        for idx, tip in X.iteritems():
            words_case_sensitive, words = self.get_words(tip, eol=True, bol=True)

            best_score = 0
            best_guess = ''

            for i in range(0, len(words)):
                guessing = words_case_sensitive[i]

                if guessing == EOL_TAG or guessing == BOL_TAG:
                    continue

                if len(guessing) < self.min_password_length:
                    continue

                words_before = list(reversed(words[:i]))
                words_after = words[i + 1:]
                score_before = self.get_score(
                    words_before, self.word_dicts_before, self.before_cutoff,
                    self.before_exponential_factor)
                score_after = self.get_score(
                    words_after, self.word_dicts_after, self.after_cutoff,
                    self.after_exponential_factor)
                score = score_before + score_after * self.after_factor

                if score > best_score:
                    best_guess = guessing
                    best_score = score

            y.append(best_guess)

        return pd.Series(y, index=X.index)

DEBUG = 356

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Load the data and extract annotated data
    df = pd.read_csv('data/' + DATA_FILE)
    data = df[(df.password.notnull()) & (df.done == '1')]

    Passwords = namedtuple('Passwords', 'data target')
    pwds = Passwords(data=data.tip, target=data.password)

    # Split into training and test set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        pwds.data, pwds.target, test_size=TEST_SET_PERCENT, random_state=0)

    parameters = {
        'before_cutoff': list(range(0, 6)),
        'after_cutoff': list(range(0, 6)),
        'before_exponential_factor': np.logspace(-1, 1, 10),
        'after_exponential_factor': np.logspace(-1, 1, 10),
        'after_factor': np.logspace(-2, 1, 20),
        'min_password_length': list(range(1, 10)),
        'margin_cutoff': list(range(0, 3)),
        'margin_factor': np.logspace(-2, 0, 10),
        'margin_exponential_factor': np.logspace(-1, 1, 10),
        'eol_factor': np.logspace(-2, 0, 10),
        'bol_factor': np.logspace(-2, 0, 10)
    }

    # Use grid search and k-fold cross validation
    clf = RandomizedSearchCV(
        PasswordEstimator(), parameters, cv=2, scoring='accuracy',
        n_jobs=multiprocessing.cpu_count() - 1, n_iter=BUDGET)

    clf = clf.fit(X_train, y_train)

    logger.info(clf.best_params_)

    test_score = clf.score(X_test, y_test)
    logger.info(test_score)

if __name__ == "__main__":
    main()
