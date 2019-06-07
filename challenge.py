import numpy as np #v1.16.2
import pandas as pd #v0.24.2
from sklearn.impute import SimpleImputer #v0.20.3
from sklearn.model_selection import train_test_split, GridSearchCV #v0.20.3
from sklearn.metrics import cohen_kappa_score, classification_report #v0.20.3
import xgboost as xgb #v0.80
import scipy.special as scp #v1.2.1
import urllib2 as url #Python standard library - python v2.7.16
from zipfile import ZipFile #Python standard library - python v2.7.16
from StringIO import StringIO #Python standard library - python v2.7.16
import datetime #Python standard library - python v2.7.16
import time #Python standard library - python v2.7.16
	
def extract_days_since_register_date(df, column):
	#Transform the register date of each order into the number of days since that date.
	#It is easier to handle a numeric value than a datetime object and this approach keeps the ordinal nature of the dates.
	df['days_since_'+column] = df[column].apply(lambda x: (datetime.datetime.now() - x).days)

def compute_features(df):
	#Create a new Data Frame in order to keep the calculated features.
	X = pd.DataFrame()

	#These columns are the same in all register of the same customer, so mean is applied just to group them.
	#Except unit_price, which is the price of a item bought by the customer.
	X[['is_churn', 'segment_code', 'group_code', 'average_unit_price']] = df.groupby('customer_code').mean().reset_index()[['is_churn', 'segment_code', 'group_code', 'unit_price']]

	#Calculate the average quantity of items by order and total price of orders for each customer.
	X[['average_quantity_by_order', 'average_total_price_by_order']] = df.groupby(['order_id', 'customer_code']).sum().groupby(level=1).mean().reset_index()[['quantity', 'total_price']]

	#Calculate the most expensive item bought by each customer.
	X['max_unit_price'] = df.groupby('customer_code').max().reset_index()['unit_price']

	#Calculate the greatest number of items and the highest price paid by each customer in a single order.
	X[['max_quantity_by_order', 'max_total_price_by_order']] = df.groupby(['order_id', 'customer_code']).sum().groupby(level=1).max().reset_index()[['quantity', 'total_price']]

	#Calculate the number of orders by customer.
	X['total_orders'] = df.groupby(['customer_code', 'order_id']).mean().reset_index().groupby('customer_code').size()

	#Calculate the number of days since last order.
	X['days_since_last_order'] = df.groupby('customer_code').min().reset_index()['days_since_register_date']

	#Calculate the number of days since the first order.
	X['days_since_first_order'] = df.groupby('customer_code').max().reset_index()['days_since_register_date']

	#Calculate the frequency of orders registered by each customer expressed as the number of orders divided by the time interval from the first to the last order.
	#Multiply by 100 just to avoid precision issues.
	X['order_frequency'] = X['total_orders'] / (X['days_since_first_order'] - X['days_since_last_order'] + 1) * 100
	
	return X

#Reading and extracting dataset.
reader = url.urlopen('https://github.com/totvslabs/datachallenge/raw/master/challenge.zip')
file = ZipFile(StringIO(reader.read()))
json_file = file.open('challenge.json')

#Creating dataframe with dataset.
df = pd.read_json(json_file)

#Preprocessing.
#Handling datetime data format.
#It is easier to handle a numeric value than a datetime object and this approach keeps the ordinal nature of the dates.
df['register_date'] = pd.to_datetime(df['register_date'], format='%Y-%m-%dT%H:%M:%SZ')
extract_days_since_register_date(df, 'register_date')

#Dropping unused variables in order to remove useless calculations in the feature extraction step.
df = df.drop(['branch_id', 'seller_code', 'item_total_price', 'register_date'], axis=1)

#Filling in missing values.
#The column 'is_churn' is used as the label of the classes (or dependent variable Y),
#	so it is a binary classification and the mean strategy would not be suitable because it would create a third class.
#The most frequent strategy is adopted due to the imbalance between the classes,
#	what makes the probability of filling in the values correctly extremely high.
#Another possibility would be using clustering on the other features of the vectors with non-missing values,
#	so as to create two cluster and, then, predict in which cluster the vectors with missing values are. This
#	information could be used as the value missing. But this would require handling the categorical features,
#	normalizing all features and finding an appropriate clutering method. And this task of finding a suitable method
#	could even involve developing a specific distance metric, since the categorical features do not lay in an Euclidean space.
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df.loc[:,:] = imputer.fit_transform(df)

#Feature extraction.
X = compute_features(df)

#Keep feature names to indentify the likely reasons later on.
feature_names = [str(name).replace('_', ' ') for name in X.columns.tolist()][1:]

#Separate features and labels.
y = X['is_churn'].to_numpy()
X = X.drop(['is_churn'], axis=1).to_numpy()

#split train and test sets, 25% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

#instantiate model
xgb_model = xgb.XGBClassifier()

#hyper parameters optimization through grid search with 5-fold cross-validation
parameters = {'nthread':[4],
			  'objective':['binary:hinge', 'binary:logistic'],
			  'learning_rate': [0.005, 0.05],
			  'max_depth': [3, 5, 7],
			  'min_child_weight': [9, 11],
			  'silent': [1],
			  'subsample': [0.6, 0.7, 0.8],
			  'colsample_bytree': [0.6, 0.7, 0.8],
			  'n_estimators': [1000],
			  'seed': [0]}
			  
clf = GridSearchCV(xgb_model, parameters, n_jobs=4, 
				   cv=5,
				   scoring=None,
				   verbose=0, refit=True)

clf.fit(X_train, y_train)

#use best model found to predict churn
y_pred = clf.predict(X_test)

#get and report likely reason
xgb_booster = clf.best_estimator_.get_booster()
contrib = xgb_booster.predict(xgb.DMatrix(X_test), pred_contribs=True)

for i, pred in enumerate(y_pred):
	if pred == 1.0:
		most_influential_reason_idx = np.argmax(contrib[i,:-1]) #the feature that most influenced positively
		client_status = 'churn'
	else:
		most_influential_reason_idx = np.argmin(contrib[i,:-1]) #the feature that most influenced negatively
		client_status = 'stay'
	print ('Customer %d will probably %s.\n\tLikely reason: %28s = %7s\t(margin contribution): %.3f') %\
		(i, client_status, feature_names[most_influential_reason_idx], '{:.3f}'.format(X_test[i, most_influential_reason_idx]), contrib[i, most_influential_reason_idx])
		
#report results
print ('\nCohen Kappa Score: %.3f\n') % (cohen_kappa_score(y_test, y_pred))
print classification_report(y_test, y_pred)