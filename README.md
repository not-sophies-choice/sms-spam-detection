# SMS Spam Detection Using Machine Learning

A machine learning project that classifies SMS messages as **spam** or **ham** using TF-IDF and Multinomial Naive Bayes.

## Overview

This project implements a supervised machine learning model for SMS spam detection using Python and scikit-learn. It uses the SMS Spam Collection dataset to learn patterns from labelled messages and predict whether a message belongs to the spam or ham category.

The workflow covers dataset exploration, data preparation, text vectorisation, model training, performance evaluation, and predictions on custom messages.

## Objectives

* Explore and prepare a labelled SMS dataset.
* Convert text messages into numerical features using TF-IDF.
* Train a Multinomial Naive Bayes classifier.
* Evaluate model performance using standard classification metrics.
* Test the trained model on custom SMS messages.

## Technologies Used

| Technology       | Purpose                                            |
| ---------------- | -------------------------------------------------- |
| Python           | Implementation                                     |
| Jupyter Notebook | Development environment                            |
| pandas           | Data loading and inspection                        |
| scikit-learn     | Feature extraction, classification, and evaluation |
| Matplotlib       | Data visualisation                                 |

## Dataset

This project uses the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) from the UCI Machine Learning Repository.

The dataset contains SMS messages labelled as:

* **Ham:** Legitimate messages.
* **Spam:** Messages labelled as spam.

The message text is the input feature, and the corresponding label is the target variable.

## Methodology

The project follows these steps:

1. **Data loading and inspection:** Load the dataset and examine its structure, missing values, and class distribution.
2. **Data preparation:** Separate message text from labels and divide the data into training and testing subsets.
3. **Feature extraction:** Use `TfidfVectorizer` to convert text into numerical features.
4. **Model training:** Train a `MultinomialNB` classifier through a scikit-learn pipeline.
5. **Model evaluation:** Calculate accuracy, precision, recall, F1-score, and the confusion matrix.
6. **Custom prediction:** Use the trained pipeline to classify user-provided messages.

## Model Performance

The following results were obtained on the held-out test set.

| Metric         | Result |
| -------------- | -----: |
| Test accuracy  | 96.05% |
| Spam precision |   1.00 |
| Spam recall    |   0.70 |
| Spam F1-score  |   0.83 |

### Confusion Matrix

| Actual Class | Predicted Ham | Predicted Spam |
| ------------ | ------------: | -------------: |
| Ham          |           966 |              0 |
| Spam         |            44 |            105 |

The test set contained 1,115 messages: 966 ham messages and 149 spam messages.

The model correctly classified 966 ham messages and 105 spam messages. It made no false-positive spam predictions but missed 44 actual spam messages.

These results demonstrate why accuracy should be considered alongside class-specific metrics, particularly recall when detecting spam.

## Getting Started

### Prerequisites

* Python 3
* Jupyter Notebook
* The SMS Spam Collection dataset

### 1. Install Dependencies

Install the required Python libraries:

```bash
pip install pandas scikit-learn matplotlib
```

### 2. Download the Dataset

Download the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) from the UCI Machine Learning Repository.

Extract the dataset and ensure that the file path in the notebook matches its actual location.

### 3. Open the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `SMS_Spam_Detection.ipynb` and run the cells in order.

Review the data inspection results, classification report, confusion matrix, and custom-message predictions.

## Repository Structure

```text
SMS-Spam-Detection/
├── README.md
├── Understand_Me.md
└── SMS_Spam_Detection.ipynb
```

* `README.md` — Project overview, methodology, results, and execution instructions.
* `Understand_Me.md` — Detailed conceptual explanations, mathematical formulas, and review questions.
* `SMS_Spam_Detection.ipynb` — Notebook containing the implementation and evaluation.

*Ensure these filenames match the files in your repository.*

## Limitations

* Performance depends on the dataset and its labelled examples.
* The model missed 44 spam messages in the selected test set.
* TF-IDF captures statistical word patterns but does not fully understand language meaning or context.
* Performance on future messages may differ from the reported results.
* The project is a notebook-based demonstration, not a deployed, real-time spam-filtering service.


## Conclusion

This project demonstrates an introductory text-classification workflow using TF-IDF and Multinomial Naive Bayes. The evaluation highlights both the model's overall performance and the importance of examining classification errors.

The implementation provides practical experience with text representation, probabilistic classification, and model evaluation.

## Documentation

For a beginner-friendly explanation of the concepts, methodology, evaluation metrics, and limitations used in this project, see [Understand Me](understand_me.md).
