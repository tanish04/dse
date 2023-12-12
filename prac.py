# Q1- programs using numpy 
#A- create 2D array . compute mean , SD ,variance 
import numpy as np
ARR1 = np.random.rand(3, 4)
mean_arr1 = np.mean(ARR1, axis=1)
std_dev_arr1 = np.std(ARR1, axis=1)
variance_arr1 = np.var(ARR1, axis=1)
print("RITIKA YADAV CSC/22/75")
print("ARR1:")
print(ARR1)
print("\nMean along axis 1:")
print(mean_arr1)
print("\nStandard Deviation along axis 1:")
print(std_dev_arr1)
print("\nVariance along axis 1:")
print(variance_arr1)
#b- 2D array of size mxn . print shape , then reshape .....
import numpy as np
m = int(input("Enter the number of rows (m): "))
n = int(input("Enter the number of columns (n): "))
original_array = np.random.randint(1, 100, size=(m, n))
print("\nOriginal Array:")
print(original_array)
print("Shape:", original_array.shape)
print("Type:", type(original_array))
print("Data Type:", original_array.dtype)
new_m = int(input("Enter the new number of rows (new_m): "))
new_n = int(input("Enter the new number of columns (new_n): "))
reshaped_array = original_array.reshape(new_m, new_n)
print("\nReshaped Array:")
print(reshaped_array)
print("Shape:", reshaped_array.shape)
print("Type:", type(reshaped_array))
print("Data Type:", reshaped_array.dtype)
#c- elements in given 1D array are zeros, non-zero, NaN... 
import numpy as np
input_array = np.array([0, 3, 5, 0, 0, 2, np.nan, 8, np.nan])
zero_indices = []
non_zero_indices = []
nan_indices = []
for i, elem in enumerate(input_array):
if np.isnan(elem):
nan_indices.append(i)
elif elem == 0:
zero_indices.append(i)
else:
non_zero_indices.append(i)
zero_indices = np.array(zero_indices)
non_zero_indices = np.array(non_zero_indices)
nan_indices = np.array(nan_indices)
print("Input Array:", input_array)
print("Indices of Zero Elements:", zero_indices)
print("Indices of Non-Zero Elements:", non_zero_indices)
print("Indices of NaN Elements:", nan_indices)
#d- three random array ....covariance, correlation .....
import numpy as np
np.random.seed(42)
size=10
Array1 = np.random.rand(size)
Array2 = np.random.rand(size)
Array3 = np.random.rand(size)
Array4 = Array3 - Array2
Array5 = 2 * Array1
covariance_1_4 = np.cov(Array1, Array4)[0, 1]
correlation_1_4 = np.corrcoef(Array1, Array4)[0, 1]
print("Covariance between Array1 and Array4:", covariance_1_4)
print("Correlation between Array1 and Array4:", correlation_1_4)
covariance_1_5 = np.cov(Array1, Array5)[0, 1]
correlation_1_5 = np.corrcoef(Array1, Array5)[0, 1]
print("Covariance between Array1 and Array5:", covariance_1_5)
print("Correlation between Array1 and Array5:", correlation_1_5)
#e- two random array .... sum of first half , product of second half..
import numpy as np
Array1 = np.random.rand(10)
Array2 = np.random.rand(10)
half_size = len(Array1) // 2
sum_first_half = np.sum(Array1[:half_size]) + np.sum(Array2[:half_size])
product_second_half = np.prod(Array1[half_size:]) * np.prod(Array2[half_size:])
print("Array1:", Array1)
print("Array2:", Array2)
print("Sum of the first half of both arrays:", sum_first_half)
print("Product of the second half of both arrays:", product_second_half)
#f- random array . size of memory occupied by array.
import numpy as np
random_array = np.random.rand(100, 100)
memory_size_bytes = random_array.nbytes
memory_size_mb = memory_size_bytes / (1024 * 1024)
print(f"Memory size of the array: {memory_size_bytes} bytes ({memory_size_mb:.2f} MB)")
#g- 2D array size mxn element in range (10,100), swap rows 
import numpy as np
m, n = 5, 4
random_array = np.random.randint(10, 100, size=(m, n))
print("Original Array:")
print(random_array)
row1, row3 = 0, 2
random_array[[row1, row3]] = random_array[[row3, row1]]
print("\nArray after swapping rows:")
print(random_array)
col_to_reverse = 1
random_array[:, col_to_reverse] = random_array[::-1, col_to_reverse]
print("\nArray after reversing a specified column:")
print(random_array)
updated_array = random_array.copy()
# Q2-
#a- series with 5 elements.sort on values and index 
import pandas as pd
data={'B':3 , 'A':1 , 'D':4 ,'E':5 ,'C':2}
my_series=pd.Series(data)
Sort_on_Index=my_series.sort_index()
print("Sorted by Index: ",Sort_on_Index)
Sort_on_Values=my_series.Sort_on_Values
print("Sorted by Values: ",Sort_on_Values)
#b- series with N elements with duplicate values. min & max ranks.
import pandas as pd
data=[3,2,1,3,4,2,5,4]
series=pd.Series(data)
#min ranks
min_ranks_first=series.rank(method='first')
#max ranks
max_ranks_max=series.rank(method='max')
#find min and max ranks
max_rank_min=max_ranks_first.min()
max_rank_max=max_ranks_max.max()
print("Min Rank:", max_rank_min)
print("Max Rank:", max_rank_max)
#c-index value of min and max element
import pandas as pd
data={'A':10,'B':20,'C':5,'D':15}
series=pd.Series(data)
#min element
min_index=series.idxmin()
print("Index of min element: ", min_index)
#max element 
max_index=series.idxmax()
print("Index of max element: ", max_index)
#Q3-----
import panda as pd 
import numpy as np 
data { = 'Column1': np.random.rand (50), 'Column2': np.random.rand (50), 'Column3': 
np.random.rand(50) }
df=pd. DataFrame(data)
null_indices = np.random.choice (df.index, size=int(0.1*len(df)), replace=False)
df.loc[null_indices]=np.nan
# A) Identify and count missing values missing values 
count= df.isnull().sum()
# B) Drop columns with more than 5 null values 
df = df.dropna (thresh=5, axis=1)
# c) Identify the row label with the maximum sum and drop that row 
row_to_drop = df.sum(axis=1).idxmax() 
df=df.drop(index=row_to_drop)
# d) Sort the DataFrame based on the first column 
df = df.sort_values(by='Column1')
# e) Remove duplicates from the first column 
df = df.drop_duplicates (subset= 'Column1')
#f) Find correlation and covariance 
correlation=df['Column1'].corr(df['Column2']) 
covariance = df[ 'Column2'].cov(df['Column3'])
# g) Discretize the second column into 5 bins 
df['Column2_bins'] = pd.cut(df['Column2'], bins=5)
print("Final DataFrame:") 
print(df)
print("\nMissing Values Count:")
print(missing_values_count)
print("\nCorrelation between Column1 and Column2:", correlation)
print("Covariance between Column2 and Column3:", covariance)
Q4-
import pandas as pd
dataset1=pd.read_excel(r"C:\Users\USER\Desktop\workshop1.xlsx")
dataset2=pd.read_excel(r"C:\Users\USER\Desktop\workshop2.xlsx")
print("Data from Workshop1")
print(" ")
print(dataset1)
print("Data from Workshop2") 
print(" ") 
print(dataset2)
#Merge the two datasets on the "Name" column 
common_attendees = pd.merge(dataset1, dataset2, on="Name", how="inner")
common_names common_attendees["Name"] 
common_names_unique=common_names.drop_duplicates() 
print("Names of students who attended both workshops: ")
print(common_names_unique)
workshop1_names = set(dataset1["Name"]) 
workshop2_names = set(dataset2["Name"])
# Find the names of students who attended only one workshop
attended_single_workshop = workshop1_names.symmetric_difference(workshop2_name)
print("Names of students who attended a single workshop only: ") 
print (attended_single_workshop) 
merged_data = pd.concat([dataset1, dataset2], inplace=True) 
total_records = len(merged_data) 
print("Total number of records in the merged data frame:", total_records)
mergea_data.set_index(['Name', 'Date'], inplace=True) 
# Generate descriptive statistics for the hierarchical data frame 
statistics=merged_data.describe() 
print(statistics)
Q5-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import mode
from scipy.stats import t
from scipy import stats
# Load Iris data
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
columns=iris['feature_names'] + ['target'])
# Display data types using pandas.info()
print("Data Types Information:")
iris_df.info()
# Check for missing values
missing_values = iris_df.isnull().sum()
print("\nNumber of missing values in each column:")
print(missing_values)
# Plot bar chart for class label frequencies
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=iris_df)
plt.title('Class Label Frequencies')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.show()
# Draw scatter plot for Petal Length vs Sepal Length with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='petal length (cm)', y='sepal length (cm)', data=iris_df, 
scatter_kws={'s':30})
plt.title('Scatter Plot: Petal Length vs Sepal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Sepal Length (cm)')
plt.show()
# Plot density distribution for Petal Width
plt.figure(figsize=(8, 6))
sns.kdeplot(iris_df['petal width (cm)'], fill=True)
plt.title('Density Distribution: Petal Width')
plt.xlabel('Petal Width (cm)')
plt.show()
# Pair plot for pairwise bivariate distribution
sns.pairplot(iris_df, hue='target')
plt.show()
# Heatmap for two numeric attributes
numeric_attributes = iris_df[['sepal length (cm)', 'sepal width (cm)']]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_attributes.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# Compute statistics for each numeric feature
numeric_stats = iris_df.describe()
print("\nDescriptive Statistics for Numeric Features:")
print(numeric_stats)
# Compute mode for each numeric feature
numeric_modes = iris_df.mode(numeric_only=True)
print("\nMode for Numeric Features:")
print(numeric_modes)
# Confidence interval and standard error for each numeric feature
confidence_interval = stats.t.interval(0.95, len(iris_df) - 1, iris_df.mean(), 
stats.sem(iris_df))
print("\nConfidence Interval for Numeric Features:")
print(confidence_interval)
# Compute correlation coefficients
correlation_matrix = iris_df.corr()
print("\nCorrelation Coefficients Matrix:")
print(correlation_matrix)
Q6-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the Titanic dataset
titanic_data = sns.load_dataset("titanic")
# a. Clean the data by dropping the column with the largest number of missing values.
missing_values = titanic_data.isnull().sum()
column_to_drop = missing_values.idxmax()
titanic_data = titanic_data.drop(column_to_drop, axis=1)
# b. Find the total number of passengers with age more than 30
passengers_over_30 = titanic_data[titanic_data['age'] > 30]
total_passengers_over_30 = passengers_over_30.shape[0]
# c. Find the total fare paid by passengers of the second class
total_fare_second_class = titanic_data[titanic_data['class'] == 'Second']['fare'].sum()
# d. Compare the number of survivors of each passenger class
survivors_by_class = titanic_data.groupby('class')['survived'].sum()
# e. Compute descriptive statistics for age attribute gender-wise
descriptive_stats_gender_age = titanic_data.groupby('sex')['age'].describe()
#f. Draw a scatter plot for passenger fare paid by Female and Male passengers separately
sns.scatterplot(x='sex', y='fare', data=titanic_data)
plt.title("Scatter Plot of Fare Paid by Gender")
plt.show()
# g. Compare density distribution for features age and passenger fare
sns.kdeplot(data=titanic_data, x="age", label="Age")
sns.kdeplot(data=titanic_data, x="fare", label="Fare")
plt.title("Density Distribution of Age and Fare")
plt.legend()
plt.show()
# h. Draw a pie chart for the three passenger classes
class_counts = titanic_data['class'].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
plt.title("Passenger Class Distribution")
plt.show()
# i. Find 5% of survived passengers for each class
survived_percent_class = titanic_data.groupby('class')['survived'].mean() * 100
print("Data cleaned by dropping the column with the most missing values:")
print(titanic_data.head())
print(f"Total number of passengers with age more than 30: {total_passengers_over_30}")
print(f"Total fare paid by passengers of the second class: {total_fare_second_class}")
print("Number of survivors by class:")
print(survivors_by_class)
print("Descriptive statistics for age attribute gender-wise:")
print(descriptive_stats_gender_age)
print("Percentage of survivors in each class:")
print(survived_percent_class)
Q7-
import pandas as pd
# Creating the DataFrame
data = {
'FamilyName': ['Shah', 'Vats', 'Vats', 'Kumar', 'Vats', 'Kumar', 'Shah', 'Shah']
'Gender': ['Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female']
'MonthlyIncome (Rs.)': [44000.00, 65000.00, 43150.00, 66500.00, 
255000.00,700000.00,550000.00,860000.00 ]
}
df = pd.DataFrame(data)
# a. Calculate and display family wise gross monthly income
family_income = df.groupby('FamilyName')['MonthlyIncome (Rs.)'].sum()
print("Family Wise Gross Monthly Income:")
print(family_income)
print()
# b. Display the highest and lowest monthly income for each family name
highest_income = df.groupby('FamilyName')['MonthlyIncome (Rs.)'].max()
lowest_income = df.groupby('FamilyName')['MonthlyIncome (Rs.)'].min()
print("Highest Monthly Income:")
print(highest_income)
print("Lowest Monthly Income:")
print(lowest_income)
print()
# c. Calculate and display monthly income of all members earning income less than Rs. 
80000.00
income_less_than_80000 = df[df['MonthlyIncome (Rs.)'] < 80000.00]['MonthlyIncome (Rs.)']
print("Monthly Income of Members Earning Less than Rs. 80000.00:")
print(income_less_than_80000)
print()
# d. Display total number of females along with their average monthly income
female_data = df[df['Gender'] == 'Female']
total_females = len(female_data)
average_income_female = female_data['MonthlyIncome (Rs.)'].mean()
print("Total Number of Females:", total_females)
print("Average Monthly Income of Females:", average_income_female)
print()
# e. Delete rows with Monthly income less than the average income of all members
average_income_all = df['MonthlyIncome (Rs.)'].mean()
df = df[df['MonthlyIncome (Rs.)'] >= average_income_all]
print("DataFrame after deleting rows with Monthly income less than average income:")
print(df)
