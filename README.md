# Password Extraction
This is a script to extract password out of natural language using foursquare comments as input. It has been tested for passwords in English but is not language dependent.

The model evaluates which word has the highest probability to be a password in a given sentence based on the popularity of words coming before and after the password.

## Training
The training set is used to build dictionaries of words' popularity at a given position before and after the password.
- The password is identified in the tip
- The tip is split into the part precending and succeeding the password

Example:
```
    tip = 'The password is FNASDIYZXC.'
    password = 'FNASDIYZXC'
    tip_before, tip_after = get_tip_before_after_password(tip, password)
    print(tip_before) # 'The password is '
    print(tip_after) # '.'
```
- Words are extracted for both parts and ordered by distance to the password.  Each tip is split into constituent words with the following considerations:
    - The ponctuation is considered as their own word.
    - Anything between double quotes is considered as a single word
    - Anything between single quotes that do not contain space is considered as a single word
    - Single quotes are otherwise considered part of other words
    - Other ASCII characters are considered as part of other words if not separated by spaces.

The beginning and end of line is added to the list.
Example:
```
    words_before = get_words('The password is ', reverse=True, bol=True)
    words_after = get_words('.', eol=True)
    print(words_before) # ['is', 'password' 'The', '_bol_']
    print(words_after) # ['.', '_eol_']
```
- The words are added to the dictionary corresponding to their position with respect to the password. 
- The value in each dictionary are normalized with respect to the greatest occurence found

Example of dictionary:
```
words_before_password = [
    # words found at distand d = 1 from the password
    {
        ':': 1,
        'password': 0.4,
        'coffee': 0.1,
        'saxophone': 0.000001,
        ...
    },
    # words found at distand d = 2 from the password
    {
        ':': 0.3,
        'password': 1,
        'wifi': 0.7
        'saxophone': 0.0000001,
        ...
    },
    ...
]

words_after_password = [
    # words found at distand d = 1 from the password
    {
        '_eol_': 1,
        'wifi': 0.4,
        ...
    },
    ...
```

## Predicting

The prediction is done by evaluating for every word the likeliness of it being a password by looking at the words preceding and succeeding it.

- The tip is split into its consituent words
```
    tip_words = self.get_words(tip, eol=True, bol=True)
    print(tip_words) # ['_bol_', 'The', 'password', 'is', 'FNASDIYZXC', '.', '_eol_']
```
- For each guess we split the words into two array of words that come before and after the password
```
    guessing = words[i] # 'FNASDIYZXC'
    words_before = list(reversed(words[:i])) # ['is', 'password' 'The', '_bol_']
    words_after = words[i + 1:] # ['.', '_eol_']
```
- We calculate the score of each array, their score is ponderated by a factor given as a parameter
```
    score_guess = score_before + score_after * self.factor_after
```
- For every word in the array we add its popularity at that position. We multiply this score by a factor depending on the distance of the word to the password.
```
    scoring_word = 'password'
    position = 1 # index is 1 so distance is 2
    score_word = words_before_password[position][scoring_word] * self.ponderate(self.exponential_factor, position)
```
- We also add the score given by the popularity of the word on both sides of the given position. This gives us the chance to give value to a word that is very popular at a different position with respect to the password than it is in the evaluated sentence.
```
    scoring_word = 'password'
    position = 1
    position_shift = -1 # we're looking at the 'left' site of the word 
    score_word += words_before_password[position + position_shift][scoring_word] \
                    * factor(position) \
                    * margin_factor(position_shift)
```

## Scoring
The score is given by the accuracy rate of the estimator.

## Parameters estimators and cross validation
The parameters are estimated using k-folded cross validation and a random parameter search.

## Results
The algorithm give a score of 92.15% on the test set with the parameters:
```
{'after_cutoff': 2,
 'after_exponential_factor': 0.774263682681127,
 'after_factor': 6.9519279617756053,
 'before_cutoff': 4,
 'before_exponential_factor': 1.2915496650148841,
 'bol_factor': 0.01,
 'eol_factor': 0.046415888336127774,
 'margin_cutoff': 1,
 'margin_exponential_factor': 10.0,
 'margin_factor': 0.016681005372000592,
 'min_password_length': 4
 }
```
