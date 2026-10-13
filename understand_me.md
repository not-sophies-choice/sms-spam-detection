# Understand Me

## SMS Spam Detection Using Machine Learning

**Project Documentation and Conceptual Guide**

**Programming Language:** Python
**Environment:** Jupyter Notebook
**Libraries:** pandas, scikit-learn, Matplotlib

---

## 1. Project Overview

SMS Spam Detection is a machine learning project that classifies text messages into two categories:

* **Spam:** Messages labelled as unwanted or suspicious.
* **Ham:** Legitimate messages that are not labelled as spam in the dataset.

The project uses **TF-IDF** to convert text messages into numerical features and a **Multinomial Naive Bayes** classifier to predict whether a message is spam or ham.

It demonstrates the basic steps of a machine learning workflow, from loading a dataset to training a model and evaluating its predictions.

## 2. Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

**Dataset source:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains SMS messages and their corresponding labels.

| Column    | Meaning                         |
| --------- | ------------------------------- |
| `label`   | The known category: spam or ham |
| `message` | The text of the SMS             |

A **record** is one message and its label.

A **feature** is the input information used to make a prediction. In our project, the message text is the input feature.

A **label** is the known category the model learns to predict. Here, the label is spam or ham.

## 3. Data Preparation

We use pandas to load and inspect the dataset.

Some useful commands are:

* `df.head()` — displays the first five rows by default.
* `df.shape` — shows the number of rows and columns.
* `df.isnull().sum()` — counts missing values.
* `df["label"].value_counts()` — counts the messages in each class.
* `df.info()` — displays information about the dataset's structure.

### Separating Inputs and Labels

```python
X = df["message"]
y = df["label"]
```

Here, `X` contains the messages and `y` contains their known labels.

### Training and Testing Sets

The dataset is divided into two subsets:

* **Training set:** Used to train the model.
* **Testing set:** Used to evaluate the trained model on messages it did not use for training.

Our implementation uses `test_size=0.20`, which reserves approximately 20% of the data for testing and 80% for training.

The parameter `random_state=42` helps make the split repeatable. The parameter `stratify=y` helps preserve the proportion of spam and ham messages in both subsets.

## 4. TF-IDF: Converting Text into Numbers

Machine learning classifiers require numerical features. TF-IDF converts words in text messages into numerical weights.

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

### Term Frequency (TF)

Term frequency measures how often a word appears in a message.

$$
TF(t,d)=f_{t,d}
$$

Where:

* \(t\) represents a term or word.
* \(d\) represents a document or SMS.
* \(f_{t,d}\) is the number of times the term appears in that message.

### Inverse Document Frequency (IDF)

Inverse document frequency measures how common or uncommon a word is across the collection of messages.

The formula used by scikit-learn's default `TfidfVectorizer` is:

$$
IDF(t)=\log\left(\frac{1+N}{1+DF(t)}\right)+1
$$

Where:

* \(N\) is the total number of documents used to fit the vectoriser.
* \(DF(t)\) is the number of documents containing the term.
* \(\log\) represents the logarithm.

A word appearing in many messages receives a lower IDF weight, while a word appearing in fewer messages receives a relatively higher weight.

### TF-IDF Formula

$$
TF\text{-}IDF(t,d)=TF(t,d)\times IDF(t)
$$

In simple terms, TF-IDF considers how often a word appears in a message and how common it is across the collection of messages.

In our project, `TfidfVectorizer()` performs this conversion and applies its default text-processing settings, including lowercasing and tokenisation.

The resulting numerical features are supplied to the classifier.

## 5. Multinomial Naive Bayes

**Naive Bayes** is a probabilistic classification method based on Bayes' theorem. It estimates the probability that an input belongs to a particular class.

Our project uses `MultinomialNB()` to classify the numerical features produced by TF-IDF.

### Bayes' Theorem

$$
P(C\mid X)=\frac{P(X\mid C)\times P(C)}{P(X)}
$$

Where:

* \(C\) represents a class, such as spam or ham.
* \(X\) represents the features of a message.
* \(P(C\mid X)\) is the probability of the class given the features.
* \(P(X\mid C)\) is the probability of observing those features given the class.
* \(P(C)\) is the prior probability of the class.
* \(P(X)\) is the overall probability of the observed features.

**How does this relate to our project?**

The classifier uses the numerical features of a message to estimate whether it belongs to the spam or ham category. It then predicts a class based on the learned probabilities.

The word *naive* refers to the simplifying assumption that features are conditionally independent given the class. Although words in real messages are not truly independent, this method can still work well for text classification.

The classifier also uses **smoothing**, which helps prevent probability estimates from becoming zero simply because a feature-class combination was not observed during training. Our `MultinomialNB()` uses its default smoothing setting.

## 6. Machine Learning Pipeline

A **pipeline** connects multiple steps into one workflow.

Our pipeline contains:

1. `TfidfVectorizer()` — converts text into numerical features.
2. `MultinomialNB()` — learns from those features and predicts the message category.

The overall workflow is:

**SMS text → TF-IDF → Naive Bayes → Predicted label**

The command `model.fit(X_train, y_train)` trains the pipeline using the training data.

The command `model.predict(X_test)` generates predictions for the test messages.

The pipeline ensures that the same fitted text transformation is used during training and prediction. It also helps prevent information from the test set from influencing the learning of TF-IDF features.

## 7. Model Evaluation

Model evaluation measures how well the predicted labels agree with the actual labels.

Our project uses accuracy, precision, recall, F1-score, and a confusion matrix.

### Accuracy

Accuracy is the proportion of all predictions that are correct.

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

Where:

* \(TP\): True positives.
* \(TN\): True negatives.
* \(FP\): False positives.
* \(FN\): False negatives.

Our model achieved **96.05% test accuracy**.

This means that approximately 96 out of every 100 test messages were classified correctly.

### Precision

Precision measures the proportion of messages predicted as positive that are actually positive.

$$
Precision=\frac{TP}{TP+FP}
$$

For spam detection, it tells us how many messages predicted as spam were labelled as spam in the test set.

Our spam precision was **1.00**, meaning there were no false-positive spam predictions in this test split.

### Recall

Recall measures the proportion of actual positive examples correctly identified.

$$
Recall=\frac{TP}{TP+FN}
$$

For spam detection, it tells us how many actual spam messages the model successfully detected.

Our spam recall was approximately **70.47%**.

This means the model detected 105 of the 149 actual spam messages and missed 44.

### F1-Score

The F1-score combines precision and recall using their harmonic mean.

$$
F_1=2\times\frac{Precision\times Recall}{Precision+Recall}
$$

Our spam F1-score was approximately **0.83**.

It gives a single measure that reflects both precision and recall.

### Support

Support is the number of actual examples belonging to each class in the test set.

* Ham: 966 messages.
* Spam: 149 messages.
* Total: 1,115 messages.

The classification report also provides two averages:

* **Macro average:** Gives equal importance to each class when averaging the metrics.
* **Weighted average:** Accounts for the number of actual examples in each class.

These averages help summarise performance across both classes.

## 8. Understanding the Confusion Matrix

A confusion matrix compares actual labels with predicted labels.

Our observed confusion matrix is:

| Actual Class | Predicted Ham | Predicted Spam |
| ------------ | ------------: | -------------: |
| Ham          |           966 |              0 |
| Spam         |            44 |            105 |

### What Do These Numbers Mean?

* **966 correctly classified ham messages:** Legitimate messages correctly predicted as ham.
* **105 correctly classified spam messages:** Spam messages correctly predicted as spam.
* **0 false positives:** No ham messages were incorrectly predicted as spam.
* **44 false negatives:** Actual spam messages were incorrectly predicted as ham.

The confusion matrix shows why accuracy alone is not enough to evaluate a model.

Although the model achieved 96.05% accuracy, it missed 44 spam messages. This means that a model can perform well overall while still making important mistakes.

We identified the number of missed messages through the confusion matrix, but we did not inspect the individual 44 messages. Therefore, we cannot claim to know the specific reasons for each misclassification.

## 9. Testing Custom Messages

The notebook includes a step where a user can enter a custom SMS message.

The trained pipeline converts the message into numerical features using the fitted TF-IDF vectoriser and then passes those features to the classifier.

The model predicts whether the message belongs to the spam or ham category.

This demonstrates how a trained model can be used to classify an individual message after training.

However, one correct prediction does not guarantee that every future message will be classified correctly.

## 10. Limitations

The project has a few important limitations:

1. **Dataset dependence:** The model learns from the examples and labels in the selected dataset.
2. **Missed spam messages:** The model missed 44 spam messages in the test set.
3. **Limited language understanding:** TF-IDF represents statistical word patterns but does not fully understand meaning or context.
4. **Changing messages:** Future spam messages may differ from those in the dataset.
5. **Limited deployment:** The project is a Jupyter Notebook demonstration, not a deployed real-time spam-filtering service.

## 11. Questions to Check Your Understanding

Try answering these questions in your own words before revisiting the relevant sections.

1. What is the difference between spam and ham?
2. Why is our project a supervised learning problem?
3. What are the input feature and target label in our dataset?
4. Why do we divide the data into training and testing sets?
5. What does TF-IDF do?
6. What is the difference between TF and IDF?
7. Write the TF-IDF formula and explain its two components.
8. What is Naive Bayes, in simple terms?
9. Write Bayes' theorem and explain what \(C\) and \(X\) represent.
10. What are the two steps in our machine learning pipeline?
11. What does accuracy measure?
12. What is the difference between precision and recall?
13. What is a false negative in our project?
14. Our model achieved 96.05% accuracy but missed 44 spam messages. What does this tell us?
15. Why should we consider more than accuracy when evaluating a classifier?

**A small challenge:** Choose five questions and try answering them without looking at the document. The goal is to explain the concepts in your own words, not memorise every definition.

## 12. Conclusion

This project demonstrates a basic machine learning workflow for SMS spam detection.

We used TF-IDF to convert text messages into numerical features and Multinomial Naive Bayes to classify them. The model achieved 96.05% test accuracy, but the confusion matrix showed that it missed 44 spam messages.

The project connects machine learning theory with practical implementation by demonstrating text representation, probabilistic classification, model training, and performance evaluation.

It also shows why interpreting a model's errors is important when judging its performance.

## A Note to the Reader

This document is designed to be a simple learning companion that introduces the fundamental concepts behind our SMS Spam Detection project. It connects the theory we learn in machine learning with the practical steps implemented in the notebook.

It covers the basics, but learning doesn't have to stop here. AI tools can be valuable companions for exploring concepts in greater depth, clarifying doubts, and discovering new perspectives.

The goal of this document is to give readers a foundation to build on, so they can ask more meaningful questions, seek deeper explanations, and engage more confidently with AI-assisted learning.

**Understand the basics. Question what you learn. Keep exploring.**
