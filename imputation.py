import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv(r'C:\Users\mobin\Desktop\Github\Dealing with missing values\melb_data.csv')

y = data["Price"]
# using only numerical predictores
melb_predictores = data.drop(["Price"], axis=1)
X = melb_predictores.select_dtypes(exclude=["object"])

# dividing data into training and validating subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# function for comparing different MAEs
def score_dataset( X_train, X_valid, y_train, y_valid ):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds=model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
# Imputation
my_imputor = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputor.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputor.transform(X_valid))

# putting back column names
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE for imputation : ")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
