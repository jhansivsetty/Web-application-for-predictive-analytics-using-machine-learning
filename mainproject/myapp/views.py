from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd  
import numpy as np  
from numpy.polynomial.polynomial import polyfit
from math import sqrt

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import RANSACRegressor, LinearRegression, TheilSenRegressor
from sklearn.linear_model import Ridge,Lasso,ElasticNet,BayesianRidge
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from django.contrib import messages


df_final = []
output = []
model_dict = {}
bestAlgorithm = ''
#the input features which have to be used for prediction
inputs_for_prediction = []
# Create your views here.
test_input = []
test_output = []

def index(request):
		return render(request, "myapp/home.html")
		
def uploadCsv(request):
	#return render(request, 'webapp/home.html')
	if "GET" == request.method:
		return render(request, 'myapp/home.html')
	else:
		csv_file = request.FILES["csv_file"]

		if not csv_file.name.endswith('.csv'):
			messages.error(request, "This is not a csv file. Please upload a csv file")
			return render(request, 'myapp/home.html')
		data_set = csv_file.read().decode('UTF-8')
		
		#lines is list of strings
		lines = data_set.split("\n")
		data_set = []
		for line in lines:
			# to remove ny extra spaces, \r and \n in the begining and end of the string
			raw_line = line.strip()
			
			#convert string to list by splitting it accoss ","
			list_line = raw_line.split(",")
			
			#print("before filtering")
			#print(list_line)
			#is any list contains empty values, it is removed
			filtered_list = list(filter(lambda a: ((a!= "")), list_line))
			
			#print("after filtering")
			#print(filtered_list)
			
			#add the filtered list to aother list 
			data_set.append(filtered_list)
			#data_set.append(list_line)
		
		#remove empty list from list of lists
		data_set = [x for x in data_set if x != []]
		
		#convert list of lists into dataframe
		df_data = pd.DataFrame(data_set)
		
		#make the 1st row as column name and remove the 1st row of the dataframe
		df_data.columns = df_data.iloc[0]
		global df_final
		df_data = df_data.drop(0)
		
		# drom al the columns with null column name 
		df_data.columns = df_data.columns.fillna('to_drop')
		if 'to_drop' in df_data.columns:
			df_data.drop('to_drop', axis = 1, inplace = True)
		try:
			df_final = df_data.astype('float')
		except ValueError:
			messages.error(request, "The docment has non numeric values in it. Please upload a valid document.")
			return render(request, 'myapp/home.html')
		#print("final data frame is ")
		#print(df_final.head(10))
		return render(request, 'myapp/home.html', {"csv_data":df_data})
		
def plotData(request):
	#creating a dataa frame
	
	df = df_final
	data_dict = {}
	reg_line_data_dict = {}
	output_col = df.iloc[:,-1].tolist()
	
	
	#convert data frame to dictionary
	for i in range((len(df.columns)-1)):
		col_data = df.iloc[:,i].values
		data_dict[df.columns[i] ] = col_data.tolist()
		
		#input_arr = np.array(col_data.astype(float))
		#output_arr = np.array(df.iloc[:,-1].astype(float))
		input_arr = np.array(col_data)
		output_arr = np.array(df.iloc[:,-1])
		
		#print("This is column data in array")
		# print(input_arr[0:5])
		# print(len(input_arr))
		# print(output_arr[0:5])
		# print(len(output_arr))
		b, m = polyfit(input_arr, output_arr, 1)
		reg_line_data_dict[df.columns[i]] = (b + m * input_arr).tolist()
	#print(data_dict)
	#to obtain only column corresponding to the output
	#output_col = df.iloc[:,-1].tolist()
	
	#to get the name of the output column
	output_col_name = df.columns[-1]
	
	#building final data object to send it to the UI
	data={
		"ip": data_dict,
		"op": output_col,
		"op_name":output_col_name,
		"reg_line_output": reg_line_data_dict
	}
	
	#print("correlations")
	#to obtain correlations
	
	#converting all the elements of dataframe to numeric
	#df_numeric = df.astype(float)
	correlations = df.corr(method = 'pearson')
	print(correlations)
	correlations.fillna(0, inplace=True)
	print(correlations)
	
	#dictionary containing correlations of each input with respect to the output
	correlations_dict = correlations.iloc[-1].to_dict()
	
	#dictionary containing correlation type of each input with respect to the output
	# ie how input is related to the output
	corr_type_dict = {}
	for key in correlations_dict:
		corr_type = findCorrelationType(correlations_dict[key])
		corr_type_dict[key] = corr_type
	#print(corr_type_dict)
	
	data["correlations"] = corr_type_dict
	return JsonResponse(data)
	
# method to find out the type of correlation based on the correlation values
# Value of r	Strength of relationship
# -1.0 to -0.5 or 1.0 to 0.5	Strong
# -0.5 to -0.3 or 0.3 to 0.5	Moderate
# -0.3 to -0.1 or 0.1 to 0.3	Weak
# -0.1 to 0.1	None or very weak
def findCorrelationType(x):
    result= { (x > -1) and (x <= -0.5) : 'Strong Negative Corelation',
              (x > -0.5) and (x <= -0.3) : 'Moderate Negative Corelation',
              (x > -0.3) and (x <= -0.1) : 'Weak Negative Corelation',
              (x > -0.1) and (x <= 0.1) : 'No Correlation',
              (x > 0.5) and (x <= 1) : 'Strong Positive Correlation',
              (x > 0.3) and (x <= 0.5) : 'Moderate Positive Correlation',
              (x > 0.1) and (x <= 0.3) : 'Weak Positive Correlation'}[1]
    return result

def trainCompleteData(request):
	#dataset = df_final.astype(float)
	dataset = df_final
	#Split the data into input and output required
	x_input = dataset.iloc[:, :-1]
	y_output = dataset.iloc[:,-1]
	
	result = predictionResult(x_input, y_output, 0.3)
	result['selectedInputFeatures'] = dataset.columns[:-1]
	global inputs_for_prediction
	inputs_for_prediction = dataset.columns[:-1]
	return render(request, "myapp/trainingOutput.html", result)

def predictionResult(x_input, y_output, testSize):
	#split the entire data into training and testing set
	x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size = testSize, random_state =0)
	test_input = x_test
	test_output = y_test
	# call the method to run the algorithms
	prediction_result = predictions_RMSE(x_train, y_train, x_test, y_test)
	final_result_rmse1 = prediction_result.rename({0:'Linear Regression',1:'Decision Tree-CART',2:'Random Forest',
                                     3:'KNN', 4:'Gradient Boosting',5:'Ada Boosting',
                                     6:'Extra Trees regressor',7:'Bag Re', 8:'Ridge', 9:'Lasso'})
	#print("accuracies with respect to each algorithm is")
	#print(final_result_rmse1)
	
	#find the algorithm which gives maximum acuracy
	maxValue = final_result_rmse1.loc[final_result_rmse1.iloc[:,0].idxmax()]
	global bestAlgorithm
	bestAlgorithm = maxValue.name
	
	result = {
		"accuracy": maxValue.values,
		"algorithm": bestAlgorithm
	}
	
	return result


def predictions_RMSE(train_input,train_output, test_input, actual_output):
	names = []
	RMSE = []
	R2 = []
	models = []
	model_dictionary = {}
	models.append(('Linear Regression', LinearRegression()))
	models.append(('Decision Tree-CART', DecisionTreeRegressor()))
	models.append(('Random Forest', RandomForestRegressor()))
	models.append(('K Nearest Neighbour(KNN)', KNeighborsRegressor()))
	models.append(('Gradient Boosting', GradientBoostingRegressor()))
	models.append(('Ada Boosting', AdaBoostRegressor()))
	models.append(('Extra Trees regressor', ExtraTreesRegressor()))
	models.append(('Bag Re', BaggingRegressor()))
	models.append(('Ridge', Ridge()))
	models.append(('Lasso', Lasso()))
	#models.append(('SVM', SVR(kernel='linear')))

	for name, model in models:
		
		model.fit(train_input, train_output)
		model_dictionary[name] = model
		predictions = model.predict(test_input)
		rmse = sqrt(mean_squared_error(predictions,np.array(actual_output)))
		r2 = r2_score(predictions,np.array(actual_output))

		names.append(name)
		RMSE.append(rmse)
		R2.append(r2)
	x = pd.DataFrame(R2)
	global model_dict
	model_dict = model_dictionary
	#x.append(names)

	#x.append(RMSE)
	# x.append(R2)
	return x
	
def trainPartialData(request):
	columns = {}
	columns["inputFeatures"] = df_final.columns[:-1]
	
	return render(request, "myapp/trainPartial.html", columns)

# training in case of selected inputs
def startPartialTraining(request):
	columns = {}
	columns["inputFeatures"] = df_final.columns[:-1]
	columns["success"] = "1"
	
	#get the values selected in the checkbox
	selected_inputs = request.POST.getlist('partialFeatures')
	columns['selectedInputFeatures'] = selected_inputs
	
	global inputs_for_prediction
	inputs_for_prediction = selected_inputs
	print("sslected values in the checkbox")
	print(selected_inputs)
	
	#filter the original dataframe to obtain on the selected columns
	new_input_df = df_final.filter(selected_inputs, axis=1)
	#print(new_input_df.head(2))
	
	output_df = df_final.iloc[:,-1]
	#print(output_df.head(2))
	
	result = predictionResult(new_input_df, output_df, 0.3)
	columns = {**columns, **result}
	#print(result)
	#print(columns)
	
	return render(request, "myapp/trainPartial.html", columns)

def predict(request):
	print("printingnnnnnnnn values input by the user")
	input_values = dict(request.GET)
	print(input_values)
	
	model = model_dict[bestAlgorithm]
	print("printing Best performing model")
	print(model)
	
	print("input features used to predict are")
	print(inputs_for_prediction)
	
	#assign the input values entered by the user to the features according to the right order which matches the training set
	inputs = []
	for input in inputs_for_prediction:
		inputs.append(input_values[input][0])
	print("ordered input values")
	print(inputs)
	
	predicted_value = model.predict(inputs)
	print("The predicted value")
	print(predicted_value)
	
	#predicted_values = model.predict(test_input)
	data={}
	data["outputValue"]=predicted_value[0]
	data["outputName"] =  df_final.columns[-1]
	#data["predictedValues"] = predicted_values 
	data["actual_output"] = test_output
	
	return JsonResponse(data)
	
# return render(request, "myapp/test.html")