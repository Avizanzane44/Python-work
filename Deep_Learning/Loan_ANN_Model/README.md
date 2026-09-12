# Loan Default Prediction using ANN (MLPClassifier)

A Deep Learning system that predicts whether a bank loan applicant has a high probability of defaulting, so the bank can assess risk before approving a loan.

## Problem Statement

A bank provides personal loans to customers, and some customers fail to repay them. This project builds a model that takes an applicant's profile and predicts the probability of default, using an Artificial Neural Network (Multi-Layer Perceptron).

## Dataset

| Feature | Description |
|---|---|
| Age | Applicant age |
| Income | Annual income |
| LoanAmount | Requested loan amount |
| CreditScore | Credit score |
| EmploymentYears | Years employed |
| ExistingLoans | Number of existing loans |
| MonthlyDebt | Existing monthly debt |
| LoanTerm | Loan duration |
| PreviousDefault | Yes/No — prior default history |
| HomeOwnership | Rent/Own/Mortgage |
| Default | Target — 0 (no default) / 1 (default) |

## Project Structure

```
├── training.py         # Loads data, preprocesses, trains the ANN, saves model + scaler
├── testing.py           # Loads the saved model + scaler and predicts on new applicants
├── Loan_Default.csv     # Dataset
├── ANN_model.pkl         # Trained MLPClassifier
├── scaler.pkl            # Fitted StandardScaler
├── requirements.txt
└── README.md
```

## Preprocessing

- `PreviousDefault`: mapped Yes/No → 1/0
- `HomeOwnership`: mapped Own/Rent/Mortgage → 0/1/2
- 70/30 train-test split (`random_state=42`)
- Features scaled with `StandardScaler`

## Model

- Algorithm: `MLPClassifier` (scikit-learn)
- Hidden layers: (22, 22, 11, 11, 5)
- Activation: tanh
- Solver: adam, adaptive learning rate
- Max iterations: 1000

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
python training.py
```
This loads `Loan_Default.csv`, trains the ANN, prints accuracy/confusion matrix/classification report, plots the training loss curve, and saves `ANN_model.pkl` and `scaler.pkl`.

**Run predictions on new applicants:**
```bash
python testing.py
```
This loads the saved model and scaler and prints a High Risk / Low Risk prediction for each sample applicant.

## Results

| Metric | Value |
|---|---|
| Training Accuracy | `_fill in after running_` |
| Testing Accuracy | `_fill in after running_` |

*(Update this table with the actual numbers printed by `training.py`.)*

## Future Improvements

- Hyperparameter tuning (grid/random search over layer sizes, activation, learning rate)
- Cross-validation instead of a single train/test split
- Handle class imbalance (e.g., class weighting or SMOTE) if the target is skewed
- Add ROC-AUC / precision-recall curve evaluation
- Deploy as a simple API or Streamlit app for interactive predictions

## Author

Add your name / GitHub profile link here.
