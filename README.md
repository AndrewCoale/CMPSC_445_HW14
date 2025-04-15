# CMPSC_445_HW14

## Execution Results:
### Logistic Regression (Bag of Words):
              precision    recall  f1-score   support

           0       0.90      0.89      0.89      5956
           1       0.95      0.97      0.96      6058
           2       0.86      0.87      0.86      5911
           3       0.88      0.87      0.87      6075
  
    accuracy                           0.90     24000
    macro avg      0.90      0.90      0.90     24000
    weighted avg   0.90      0.90      0.90     24000


### Support Vector Machine (TF-IDF):
              precision    recall  f1-score   support

           0       0.92      0.90      0.91      5956
           1       0.95      0.98      0.97      6058
           2       0.87      0.88      0.88      5911
           3       0.90      0.88      0.89      6075

    accuracy                           0.91     24000
    macro avg      0.91      0.91      0.91     24000
    weighted avg   0.91      0.91      0.91     24000

## Preprocessing:
- Changed the Class Index column from the range 1-4 to 0-3
- Merged the Title and Description columns
- LR: Used CountVectorizer to create a bag of words with up to 10,000 features and without English stopwords
- SVM: Used TfidfVectorizer to represent the text as TF-IDF scores with the same stopwords and max feature limit

## Performance Discussion:
- Both models perform pretty well, but SVM manages to slightly outperform LR in every metric. Its strength comes from term weighting in order to leverage the importance of distinctive words, making it better at classifying nuanced content. Roughly 90% accuracy is nothing to complain about in either case, though.
