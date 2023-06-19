# Analyzing and Predicting Graduation Rate of Universities

## Executive Summary
   Education is the key to progress, innovation and to building the roadblocks for the future of a society. Choosing the right destination for knowledge is crucial for both students - to accumulate their desired skillset - as well as universities - to impart their curated wisdom. But often, several factors come into play that hinders students from attainting their full potential and universities, from providing the best educational experience. Consequently, a society fails to reach the pinnacle of it’s prosperity. Our research aims to shed light into the factors that come into play in determining the graduation rate of a university. Understanding the significance of one or more factors that might influence graduation rates of a university will be beneficial for an educational institution to decide on areas of improvement. Additionally, identifying the relative importance of said factors would help the universities in determining how to allocate its resources on developing the areas lacking. This information is valuable for universities seeking to improve their educational facilities and subsequently providing students with the education they deserve. It will also be beneficial in assisting students select the univeristy most suited in fulfilling their aspirations. 

## Statement of Scope
### Project Objectives
This research aims to identify leading factors that influence the graduation
rate of a university. Our study is built upon a dataset comprised of various
attributes pertaining to educational measures, collected from 777 universities
in the US in the year of 2019. Our research will focus on answering the
following:

-   Predicting graduation rate of universities to assist students in selecting
    their desired destination of study

-   Identifying which factors affect a university’s graduation rate

-   Determining relative importance of factors affecting a university’s
    graduation rate so it’s resources can be focused on improving those areas
    
### Unit of Analysis
Our unit of analysis for this project will be a College, since it identifies the university/educational institution for which the graduation rate is measured. This unit uniquely identifies every record in our dataset and hence, further levels of analysis aren’t required. 
#### Independent and Dependent Variables 
The target variable for this project will be the grad_rate, which describes the graduation percentage of a particular university. Since the leading determinant of the success of a university or educational institution is it’s students completion of its curriculum, the use of this measure as our target variable is justified. Conversely, graduation rate is objectively the definitive factor for a student to choose their university. 
We take several independent variables into account that include various numeric measures and a single categorical flag. The categorical flag variables identify the university as either a public or private university. The numeric measures span from enrollment data like number of university applications received, accepted, students enrolled to institutional measures like cost of tuition, faculty credentials, part-time/full-time enrollment count etc.

## Project Schedule
We estimate the project to take a total of 12 weeks to be completed. The team will meet every week to discuss progress, clear roadblocks and assess the task roadmap to achieve the project’s objectives. The tasks will be taken up and completed by pairs of team members, in the spirit of collaboration and efficiency; members with similar skills suited for each task will work together.<br />
Outlined below are the two major milestones of the project –

-   Project Deliverable 1
-   Project Deliverable 2<br />
Following is the Gantt Chart:<br />
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/Gantt_Chart.jpg)

## Data Preparation
The most important part of a machine learning project is to process the raw data, transform it in the required format and prepare it so that machine learning algorithms can be run on it to uncover insights and make predictions.<br />
-   Data was downloaded from Kaggle and can be accessed at: [College Data](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/data/College_Data.csv)
-   Python file used for Data Preparation:[Data Preparation](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/code/Data_Preparation.ipynb)
-   Final dataset generated after above code: [College Data Transformed](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/data/College_Data_Transformed.csv)<br />

Using the dataset from Kaggle, we started with analyzing the datatypes of columns and made changes, if required, as per their use and for memory optimization. As a good coding practice, we then updated the column names to meaningful names. Also, we prepared datasets for descriptive analysis by creating separate tables for categorical and numerical columns. We then checked the dataset for duplicate rows, identified primary key, looked for missing values in any column and took appropriate action on them, eliminated the rows with erroneous or invalid column values, analyzed the outliers and took the measures based on domain and data knowledge.
As a part of data transformation, we created new columns as per our requirement by performing mathematical operations on existing columns. Also, some of the columns were aggregated to form a single column. Furthermore, all the variables were normalized i.e., were converted to same scale so that the machine learning model does not get biased, and we are able to find true order of variables based on their importance. Finally, we checked if our final columns could be reduced using PCA to lesser columns.
All the above-mentioned steps are explained in detail in following sections: 

### Data Access
The dataset for this project was obtained from Kaggle.com (https://www.kaggle.com/yashgpt/us-college-data). Kaggle is a subsidiary of Google LLC and it is an online community of data scientists and machine learning practitioners. We believed this would be the perfect destination to procure our dataset, as it provides us a platform for discussion and eventually publishing of our project. The dataset was contributed by Kaggle user Yash Gupta (https://www.kaggle.com/yashgpt) and contains information pertaining to various university measures that might affect its graduation rate. 

### Data Cleaning
As the first step of our analysis, we explored the data to understand the nature
and distribution of the variables. Some of the steps performed to prepare it for
transformation include –

-   Checking the data type of each column

    The data types of variables were assessed to ensure the appropriate
    assignment for the data it measured. An incorrect assignment of numeric
    column as text would lead to problems in analysis. Additionally, we want to
    optimize the memory usage by assigning relevant datatypes for each numerical
    variable, for example, assigning dtype as int16 in place of int64 if the
    values were 1- or 2-digit values.

The datatypes numeric and categorical were assigned as expected, and therefore,
modification of datatype was not required. We changed the datatype of columns
which had percentage values from int64 to int16 as it only included values in
the range of 0-100.
```
college_data['pct_phd_faculty'] = college_data['pct_phd_faculty'].astype('int16')
```
#### Before transformation:
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/1.jpg)
#### After transformation:
After modifying the datatype, the memory usage reduced from 115.5 KB to 88.4KB
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/2.jpg)<br />

-   Renaming columns to interpretation-friendly names

    Some of the column names were found to be non-descriptive of the data it
    measured and hence, they were renamed to make them more user-friendly and
    easy to interpret.

    Some examples include *Top10Perc* to *PctTop10HS* (describes the percentage
    of enrolled students who were in the Top 10% in their respective High
    School), *Unnamed* to *college_name* (describes the college for which the
    data was collected) and *Expend* to *exp_per_student* (describes the
    expenditure that the college spends on one student for instructional
    facilities)
```
#Renaming columns
new_column_names = {'Unnamed: 0':'college_name', 'Private':'is_private', 'Top10perc':'PctTop10HS','Top25perc':'PctTop25HS','Apps':'no_of_applications', 'Accept':'no_of_apps_accepted', 'Enroll':'no_of_enrolled', 'F.Undergrad': 'no_of_FT', 'P.Undergrad': 'no_of_PT', 'Outstate':'outstate_tuition', 'Room.Board':'room_board_cost', 'Books':'books_cost', 'Personal': 'personal_spending', 'PhD': 'pct_phd_faculty', 'Terminal':'pct_termianl_faculty','S.F.Ratio':'sf_ratio', 'perc.alumni':'pct_alumni_donors','Expend':'exp_per_student', 'Grad.Rate':'grad_rate'} 
 
college_data.rename(columns=new_column_names, inplace=True)
```
-   Grouping numeric and non numeric columns for ease of descriptive analysis

    Two new data frames were created to segregate the numeric and non-numeric
    columns. This would be useful in later stages while performing operations
    specific to each data-type.

```
#Identifying numeric and non-numeric columns
cat_df = college_data.select_dtypes(include=['object'])
num_df = college_data.select_dtypes(exclude=['object'])
 
numeric_columns = []
categ_columns = []
def segregateBasedOnColumnTypes(non_numeric_df, numeric_df):
    for col in non_numeric_df:
        categ_columns.append(f"{col}")
    for col in numeric_df:
        numeric_columns.append(f"{col}")
 
        
segregateBasedOnColumnTypes(cat_df, num_df)
```
-   Assigning an index/unique identifier for each row the data set

    To uniquely identify each row in our dataset, the variable college_name
    (describes the name of the College for which the data was collected) was
    determined to be the ideal candidate. Prior to assigning it as our index, it
    was evaluated for the presence of duplicate rows to confirm its uniqueness.
```
#Checking if college_name has any duplicates
college_data['college_name'].duplicated().any()
```
-   Finding the presence of missing values

    Missing values in a dataset would result in several problems - they would
    produce incorrect results for analysis, skew the distributions for data and
    further make interpretations of modelling results cumbersome.

    To avoid this issue, all the variables were checked for the presence of
    missing values. The data frame was evaluated using isnull() function, and
    all the columns returned a Boolean value of False indicating that there were
    no missing values in the dataset.

    Since, no missing values were found, further steps to handle missing values
    were not necessary.
    <br />
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/3.jpg)<br />
-   Checking for presence of duplicate rows

    Errors in data collection and gathering is commonly observed and would serve
    to introduce redundancy while also skewing it’s distribution. To avoid this
    issue, the data set was checked for the presence of duplicate rows.

    Function duplicated() was used to check if there were any duplicate entries
    in the dataset, and a Boolean value of False was returned indicating that
    there no duplicate values. Therefore, we did not have to drop any duplicate
    rows.<br />
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/4.jpg)

-   Eliminating erroneous values

    Often prior to analysis, a basic summary of data that describes the mean,
    std, min and max values of numeric columns is prepared to spot odd or
    erroneous values. Utilizing the summary(), we analyzed the distribution of
    data for each column. The summary of our dataset revealed a few oddities in
    certain variables –

    Looking at the max value for *pct_phd_faculty*(the percentage of college
    faculty who had PhD’s), it observed a value of 103; this is incorrect since
    a maximum measured percentage value can only be 100. Similarly, the oddity
    was observed in *grad_rate* (the target variable that measured graduation
    percentage of each college). There was a total of 2 rows which had such
    erroneous values.

    Our data frame was filtered to include only values which did not exceed 100
    for columns that contained percentage values.
```
## Removing erraneous data
 
# Count the number of rows which have graduation rate greater than 100%
print('No. of rows with erraneous graduation rate: ' + str(len(college_data[college_data['grad_rate'] > 100])))
# Retain only rows without erranous data
college_data = college_data[college_data['grad_rate'] < 100]
```
-   Outlier analysis<br />
**Categorical Variables:**
A bar chart is plotted for the categorical variable ‘is_private’ to compare the distribution of data among private and public categories. There are around 550 private colleges in comparison to a little more than 200 public colleges. The distribution is not very disproportionate and therefore, we conclude that this variable does not contain outliers.<br />
```
plt.figure(figsize=(10, 6))
college_data['is_private'].value_counts().plot.bar()
```
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/5.png)

**Numeric Variables:**
Outliers in numerical variables can be identified using boxplot diagrams of each variable. Datapoints outside the whiskers are considered as outliers.
Almost all the variables contain some number of outliers. For example, variable expenditure per student contains a large number of outliers, while out of state tuition has only one outlier.
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/12.png)<br />
To better understand the number of outliers present in the data, rows containing outliers are identified using Interquartile Range and stored in a separate dataframe.
A value can be classified as an outlier if the value is greater than 1.5 *IQR (Interquartile Range =   75th percentile - 25th percentile)
```
# calculate Q1 and Q3
Q1 = college_data.quantile(0.25)
Q3 = college_data.quantile(0.75)
 
# calculate the IQR
IQR = Q3 - Q1
 
# filter the dataset with the IQR
IQR_outliers = college_data[((college_data < (Q1 - 1.5 * IQR)) |(college_data > (Q3 + 1.5 * IQR))).any(axis=1)]
IQR_outliers
```
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/13.png)<br />
We find that a significant portion of our dataset lies outside the IQR.
While we conducted the above analysis to identify outliers these methods are not very effective in pinpointing individual records for our business case, as each college have their own policy and philosophy when it comes to majority of the dependent variables. Hence to look at a holistic view of each college to identify outliers Cook’s distance method was employed.
To find Cooks distance for each college we first fit a basic linear model to identify records that are skewing the model and have high weights.<br />
**R Code**
```

mod <- lm(accept_rate ~ no_of_applications+no_of_apps_accepted+no_of_enrolled+PctTop10HS+PctTop25HS+no_of_FT+no_of_PT+outstate_tuition+room_board_cost+books_cost+personal_spending+pct_phd_faculty+pct_termianl_faculty+sf_ratio+pct_alumni_donors+exp_per_student+grad_rate+total_misc_cost, data = reduction_data)
Next we calculate cooks distance and plot it based on record number.
cooksd <- cooks.distance(mod)
df<-as.data.frame(cooksd)
sample_size <- nrow(reduction_data)
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4/sample_size, names(cooksd),""), col="red")  # add labels
```
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/14.png)<br />
We use the Cook's distance>4/n metric to identify records that cross that the threshold. We find that one college has highly skewed values, the college record was identified to be Rutgers at New Brunswick college which was dropped as we can afford to lose a single record.


### Data Transformation
Following are the steps we took towards data transformation -

-   Creating new variables

    In order to understand the popularity and admission competitiveness of each
    college, we calculated the acceptance rate based on the number of
    applications and the number of application that were accepted.

    Acceptance rate = (no. Of accepted / no of applications) \* 100
```
# Calculate acceptance rate of each college based on the number of applicants and the number of applications accepted by the college 
college_data["accept_rate"] =  round((college_data["no_of_apps_accepted"] / college_data["no_of_applications"])*100, 2) 
```
-   Aggregating columns

    We calculated the total miscellaneous costs by aggregating (sum) the cost
    columns – cost of books, cost of room boarding, and personal expenses. We
    have not included out-of-state tuition fee (cost variable) in the
    calculation as we do not have data on the in-state tuition fee. For this
    reason, we have excluded out-of-state tuition fee from our expense analysis.
```
## Aggregation of columns  
def getTotalMiscCost(df):
    	overallCost = []
    
#For each column, get the college's overall costs
    for index, row in df.iterrows():
        overallCost.append(row['room_board_cost'] + row['books_cost'] 
        + row['personal_spending']) 
        
    return overallCost
```
-   Normalizing variables

    Since we are preparing the data for machine learning models, we need to make
    sure that our dataset is normalized i.e., all the numeric columns in the
    dataset are changed to a common scale, without distorting differences in the
    ranges of values. This step is important because we do not want to induce
    bias in our machine learning model, where columns with larger value may come
    up as important even when they are not.

    Therefore, we normalized our dataset to a scale between 0 and 1:
```
#Data Normalization

#Copying the dataframe to a new variable
df = college_data

#Removing categorical variable since it can't be normalized
df = df.drop(['is_private'],axis=1)

#Creating a normalized dataframe where each value is substracted by mean of its column and divided by standard deviation of its column
#Note: This dataframe will only be used for Machine Learning Model and not for descriptive statistics
normalized_df=(df-df.min())/(df.max()-df.min())
normalized_df.head()
```
Below is a screenshot of how the normalized data looks like:<br />
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/11.png)<br />
We won’t be using this dataset for descriptive analysis. We have prepared it only for training machine learning models.

### Data Reduction
A principal component analysis and Factor Analysis was done to find out principal factors and reduce the data. Each of the analysis below was performed on a mutually exclusive sample of data to ensure validity.
-   Principal Component Analysis <br />
The scree plot below signifies 3 components which can be used for modelling<br />
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/6.png)<br />
-   Factor Analysis <br />
Through trial and error we found that 4 factors ensures maximum amount of variables to be included in atleast one of the factors. But we notice certain variables (highlighted in the below image) which do not contribute enough variance to either of the factors. <br />
![alt text](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/assets/factoranalysis.png)<br />
While we performed PCA and factor analysis to find potential avenues for data reduction, our dataset is not large enough to warrant such an information loss. In addition, our business case intends to provide interpretability from our modelling to help colleges find avenues to improve their graduation rate. Hence we have decided not to go ahead with data reduction. <br />
The code for the above analysis can be found at [DataReduction.R](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/code/datareduction.r)
### Data Consolidation
We are using a single data file for our analysis. Our independent and dependent variables are stored in the same dataframe. Also, the new variables that we created were added to the same data file. Therefore, we didn’t have to undertake any data consolidation steps.

### Data Dictionary
Following is the data dictionary of our final clean and transformed dataset that we are using in the analysis. The data can be found at [College_Data_Tansformed](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-1-cobra-kai/blob/main/data/College_Data_Transformed.csv)

| Name                  | Description                                                                                                                                                                                           | Data Type  | Size  | Example                      |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------|------------------------------|
| college_name          | Name of the college<br />  *Purpose: serves as unique identifier for each row*                                                                                                                              | Object     | 2     | Abilene Christian University |
| is_private            | A factor with levels No and Yes indicating private or public university<br />  *Purpose: Serves as a categorical variables to predict differences in graduation rates for private and public universities*  | Object     | 2     | Yes                          |
| no_of_application     | Number of applications received<br />  *Purpose: Will serve as a measure of university demand*                                                                                                              | Integer    | 64    | 1660                         |
| no_of_apps_accepted   | Number of applications accepted<br />  *Purpose: Will serve as a measure of the university’s selectivity in students*                                                                                       | Integer    | 64    | 1232                         |
| no_of_enrolled        | Enroll Number of new students enrolled<br />  *Purpose: Will serve as a measure of accept vs enroll ratio*                                                                                                  | Integer    | 64    | 721                          |
| PctTop10HS            | Pct. new students from top 10% of H.S. class<br />  *Purpose: Signifies the academic quality of enrolled students*                                                                                          | Integer    | 16    | 23                           |
| PctTop25HS            | Pct. new students from top 25% of H.S. class<br />  *Purpose: Signifies the academic quality of enrolled students*                                                                                          | Integer    | 16    | 52                           |
| no_of_FT              | Number of fulltime undergraduates<br />  *Purpose: Will be used to identify impact of graduation among fulltime and parttime students*                                                                      | Integer    | 64    | 2885                         |
| no_of_PT              | Number of parttime undergraduates<br />  *Purpose: Will be used to identify impact of graduation among fulltime and parttime students*                                                                      | Integer    | 64    | 537                          |
| outstate_tuition      | Out-of-state tuition<br />  *Purpose: Will be used to determine whether cost of education is significant in dropouts and consequently lower graduation rates*                                               | Integer    | 64    | 7440                         |
| room_board_cost       | Room and board costs<br />  *Purpose: Will be used to determine whether housing costs are significant in affecting graduation rate*                                                                         | Integer    | 64    | 3300                         |
| books_cost            | Estimated book costs<br />  *Purpose: Will be used to determine whether cost of education is significant in dropouts and consequently lower graduation rates*                                               | Integer    | 64    | 450                          |
| personal_spending     | Estimated personal spending<br />  *Purpose: Will be used to determine whether cost of living is significant in affecting graduation rate*                                                                  | Integer    | 64    | 2200                         |
| pct_pht_faculty       | Percentage of faculty with Ph.D.’s<br />  *Purpose: Will be used to determine whether the education level of faculty affects graduation rate*                                                               | Integer    | 16    | 70                           |
| pct_termianl_faculty  | Percentage of faculty with terminal degree<br />  *Purpose: Will be used to determine whether the education level of faculty affects graduation rate*                                                       | Integer    | 16    | 78                           |
| sf_ratio              | Student to faculty ratio<br />  *Purpose: Will be used to determine whether the population of a classroom affects graduation rate*                                                                          | Float      | 64    | 18.1                         |
| pct_alumni_donors     | Pct. alumni who donate<br />  *Purpose: Will be used to determine whether an active and loyal alumni community affects graduation rate*                                                                     | Integer    | 16    | 12                           |
| exp_per_student       | Instructional expenditure per student<br />  *Purpose: Will be used to determine whether the instructional fees of a college affects graduation rate*                                                       | Integer    | 64    | 7041                         |
| grad_rate             | Graduation rate<br />  *Purpose: The target variable being measured*                                                                                                                                        | Integer    | 16    | 60                           |
| total_misc_cost       | Aggregated sum of costs of books, room boarding and personal student expenses<br /> *Purpose: Will serve as a measure of a student’s expenses to study at a particular university*                          | Integer    | 64    | 5950                         |
| accept_rate           | Acceptance percentage of a university<br /> *Purpose: Will serve as a measure of a university’s selectivity in students*                                                                                    | Integer    | 16    | 74.22                        |



## Descriptive Statistics and Analysis
1\. Starting with the simple summary of all the columns, we have created
following tables that describes Mean, Median, Standard Deviation, Minimum Value
and Maximum Value of all columns.<br />
| Variable Name        | Mean     | Median | Standard Deviations | Minimum | Maximum |
|----------------------|----------|--------|---------------------|---------|---------|
| no_of_applications   | 2997.65  | 1557   | 3879.23             | 81      | 48094   |
| no_of_apps_accepted  | 2029.57  | 1110   | 2468.83             | 72      | 26330   |
| no_of_enrolled       | 783.99   | 434    | 935.82              | 35      | 6392    |
| PctTop10HS           | 27.27    | 23     | 17.2                | 1       | 96      |
| PctTop25HS           | 55.55    | 54     | 19.64               | 9       | 100     |
| no_of_FT             | 3726.12  | 1707   | 4885.91             | 139     | 31643   |
| no_of_PT             | 863.76   | 363    | 1532.88             | 1       | 21836   |
| outstate_tuition     | 10406.13 | 9985   | 3987.69             | 2340    | 21700   |
| room_board_cost      | 4347.76  | 4191   | 1089.5              | 1780    | 8124    |
| books_cost           | 548.66   | 500    | 164.9               | 96      | 2340    |
| personal_spending    | 1342.68  | 1200   | 671.41              | 250     | 6800    |
| pct_phd_faculty      | 72.58    | 75     | 16.15               | 8       | 99      |
| pct_termianl_faculty | 79.67    | 82     | 14.62               | 24      | 100     |
| sf_ratio             | 14.09    | 13.6   | 3.95                | 2.5     | 39.8    |
| pct_alumni_donors    | 22.61    | 21     | 12.24               | 0       | 64      |
| exp_per_student      | 9605.26  | 8377   | 5111.39             | 3186    | 56233   |
| grad_rate            | 64.93    | 65     | 16.72               | 10      | 99      |
| total_misc_cost      | 6239.11  | 6100   | 1197.99             | 3470    | 12330   |
| accept_rate          | 74.99    | 77.93  | 14.36               | 15.45   | 100     |

Discussing a few important columns and their statistics:

-   Acceptance Rate (No. Of Applications, No. Of Applications Accepted) has a
    wide range that tells us that are our dataset consists of all kinds of
    colleges ranging from very competitive to easy-admit colleges

-   The standard deviation of PctTop10HS and PctTop25HS is around 17 and 19
    percent respectively, which refers to the average difference between
    (quantitative measure of ) the quality of students among different colleges

-   Number of Fulltime Undergraduates has a very high standard deviation. A
    reason for that could be that only a few colleges might have executive
    programs that admits working individuals

-   The mean of Outstate Tuition fees is around \$10,000 while its standard
    deviation is \$4000, resulting in a range between \$2300 and \$21000. This
    shows that there is a huge difference in the Outstate Tuition fees of
    different colleges.
    
#### Scope of Predictive Modelling: 
We aim to predict the graduation rate of universities using various other independent variables present in our dataset. Since most of the variables in our dataset are continuous variables, regression models appear to be the best direction to head in. Additionally, we also plan on performing unsupervised clustering of universities and testing different models on each cluster. The models will be tested on defined metrics and the best model will be selected for implementation.

## Deliverable 1: Conclusion and Discussion

We conducted data cleaning and transformation to generate useful features for our modelling purposes. In addition, we performed outlier analysis to handle records that would potentially skew our model. After completing data preprocessing we conducted descriptive analysis to understand the relationship of among variables and their respective spread. Due to the nature of our dataset we were bound to get a large spread on the data - as each college would have different sets of variables according to their respective policies towards their students and faculty - along with other external factors such as location. Our aim is to consolidate these varied colleges into a single model and understand how their policies affect graduation rate. <br />

On identifying important factors that affect graduation rate we can in turn help colleges focus their resources in areas that would yield the highest results. Since resources that the colleges have are limited, identifying such avenues is paramount to gain a competitive edge and to help students achieve their goals.



# **The Second Deliverable**

## **Visualizations**

### **Correlation among variables**

A heatmap is used to understand the correlation between the variables in the dataset. In the plot, a very light shade (almost beige) indicates a strong positive correlation, while a dark shade (deep purple) indicates a strong negative correlation. It appears that variable &#39;Acceptance Rate&#39; has a negative correlation with almost all the variables in the dataset. This explains that with an increase in variables such as number of applicants, top25students, the acceptance rate is very low. Therefore, it makes sense that acceptance rate has a negative correlation with these variables, for popular colleges with numerous applicants and competitive students the acceptance rate is low.

Similarly, some of the strong positive correlations are for number of applications, number of applicants accepted and number of enrolled students.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/15.png)

### **Comparison of tuition and other expenses with graduation rate**

We would like to explore the relationship between the financial aspect of a college and it is graduation rate. To understand this, a scatter plot is created to visualize the impact of cost on graduation rate. It appears that as the tuition cost increases, the graduation rate also linearly increases. However, miscellaneous cost does not appear to have a significant relationship with the graduation rate. For most colleges, the miscellaneous cost such as room rent, books and so on ranges between $5,000 - $7,000.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/16.png)


### **Relationship between top-performing high school students and graduation rate**

We plot a bar chart to understand if there is an existing relationship between the graduation rate and top performing high school students. The dataset included information on the percentage of top 10 and top 25 high school students that each college enrolled. From the below visualizations, we can see that as the percentage of top-performing students increases, the graduation rate also increases. Therefore, the caliber of a student plays a role in the overall graduation rate of a college.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/17.png)


![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/18.png)


### **Relationship between Graduation rate with Percentage of PhD faculty and type of college**

To understand the relationship between graduation rate and the faculties present in a university, we create two scatter plot – private and public colleges. With an increase in the number of PhD faculty, the graduation rate also increases for a college. This behavior is more pronounced in private colleges, probably since private colleges can afford to recruit high-salary PhD faculty, thereby increasing the quality of education.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/19.png)


Further, to understand the distribution of faculty (PhD and Terminal) in private and public colleges a bar chart is utilized. In comparison to non-private colleges, private colleges have a lower percentage of PhD and Terminal faculty. In general, % of PhD faculty are lower than the % of terminal faculty in both private and non-private colleges.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/20.png)


From the above visualizations, we can conclude that the some of the major features that could influence modelling are the financial data such as outstate tuition, performance of students (% of top 10 and top 25 high school students), and the education level of the faculty.

The Jupyter Notebook with the code and output for the visualizations can be found here: [Visualization](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-2-cobra-kai/blob/main/code/Plot2.ipynb)

## **Modeling Techniques Selection**

To arrive at an overall optimum model, we have tried varied techniques and compared them. As mentioned before, our overall goal is to predict Graduation rates based on factors such as Cost, Acceptance Rate, etc. Apart from the prediction model we also aim to determine variables that are important predictors of graduation rates. As Graduation Rate is a numeric variable, we employ regression techniques for our analysis.

### **1. Multiple Regression:**

While multiple regression is a simpler model compared than the other techniques, it has proven to be very useful for interpreting variables and their relationships to the dependent variable.

The model has the general linear regression assumptions about the data including:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A) Homoscedasticity

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B) No multicollinearity

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C) Residual errors have a normal distribution

These assumptions are tested before developing multiple regression models.

### **2. Random Forest Regressor**

While the interpretability of this model is not as high as Multiple regression, random forest reduces bias in our data, it can explain complicated relationships between variables better than multiple regression which can only show linear relatonships. Although it&#39;s possible random forest might overfit the data. Random Forest has no formal distributional assumptions about the data

### **3. Neural Networks**

As with Random Forest, Neural Networks can better explain complicated relationships. Neural Networks also have virtually no interpretability compared to other models. Neural networks have no assumptions about the data

### **4. Regression Tree**

This model is useful for our key aim to determine important factors affecting graduation rate. Regression trees don&#39;t have any major assumptions about the data. Moreover, the model is also very sensitive to the scale of the predictor variables.

## **Data Splitting and Sub-Sampling**

We have about 760 records available in our data for analysis. While the data is large enough to not require subsampling, our models might be hampered by the lack of training data if we go ahead with a validation set. Hence, we would be moving forward with a 70/30 train-test split as it would provide ample training data and representative test data

We employ stratified sampling method (process of splitting the data while ensuring one of the variables maintains their frequency distribution) to make sure our test and training data are representative of the sample. According to our initial analysis the acceptance rate of a college was highly correlated with the graduation rate. Hence, we use acceptance rate as a basis for stratification. As the technique requires categorical variables to stratify, we binned acceptance rate into 6 quantiles and used that to create a split

To validate our split we calculated mean, median and standard deviation the variables in the train and test set.

| Variable | Overall mean | Test mean | Train mean | Overall Median | Test Median | Train Median | Overall Std dev | Test Std Dev | Train Std Dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no\_of\_applications | 0.06 | 0.06 | 0.06 | 0.03 | 0.03 | 0.03 | 0.08 | 0.07 | 0.08 |
| no\_of\_apps\_accepted | 0.07 | 0.07 | 0.07 | 0.04 | 0.04 | 0.04 | 0.09 | 0.08 | 0.10 |
| no\_of\_enrolled | 0.12 | 0.12 | 0.12 | 0.06 | 0.06 | 0.06 | 0.15 | 0.14 | 0.15 |
| PctTop10HS | 0.28 | 0.29 | 0.27 | 0.23 | 0.24 | 0.23 | 0.18 | 0.19 | 0.18 |
| PctTop25HS | 0.51 | 0.52 | 0.51 | 0.49 | 0.52 | 0.49 | 0.22 | 0.23 | 0.21 |
| no\_of\_FT | 0.11 | 0.11 | 0.11 | 0.05 | 0.05 | 0.05 | 0.16 | 0.16 | 0.15 |
| no\_of\_PT | 0.04 | 0.04 | 0.04 | 0.02 | 0.02 | 0.02 | 0.07 | 0.07 | 0.07 |
| outstate\_tuition | 0.42 | 0.42 | 0.42 | 0.39 | 0.41 | 0.39 | 0.21 | 0.21 | 0.21 |
| room\_board\_cost | 0.40 | 0.41 | 0.40 | 0.38 | 0.38 | 0.38 | 0.17 | 0.18 | 0.17 |
| books\_cost | 0.20 | 0.20 | 0.20 | 0.18 | 0.19 | 0.18 | 0.07 | 0.07 | 0.07 |
| personal\_spending | 0.17 | 0.17 | 0.17 | 0.15 | 0.15 | 0.15 | 0.10 | 0.11 | 0.10 |
| pct\_phd\_faculty | 0.71 | 0.71 | 0.71 | 0.74 | 0.74 | 0.74 | 0.18 | 0.18 | 0.18 |
| pct\_termianl\_faculty | 0.73 | 0.74 | 0.73 | 0.76 | 0.76 | 0.76 | 0.19 | 0.19 | 0.19 |
| sf\_ratio | 0.31 | 0.31 | 0.31 | 0.30 | 0.29 | 0.30 | 0.11 | 0.11 | 0.10 |
| pct\_alumni\_donors | 0.35 | 0.35 | 0.36 | 0.33 | 0.33 | 0.33 | 0.19 | 0.18 | 0.20 |
| exp\_per\_student | 0.12 | 0.13 | 0.12 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.09 |
| total\_misc\_cost | 0.31 | 0.32 | 0.31 | 0.30 | 0.28 | 0.30 | 0.14 | 0.14 | 0.13 |
| accept\_rate | 0.70 | 0.70 | 0.71 | 0.74 | 0.74 | 0.74 | 0.17 | 0.18 | 0.17 |
| grad\_rate\_round | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.19 | 0.18 | 0.19 |

All variables have a comparatively equal split because of stratified sampling. To statistically validate the split, we conducted t-tests on the variable (PctTop25HS) with the biggest difference in mean and median as highlighted above.

**Between Train set and overall set** :

t-statistic: 0.41 p-value: 0.68

**Between Test set and overall set** :

t-statistic: 0.66 p-value: 0.50

The difference is statistically insignificant

As all variables are normalized and have a smaller difference in means between samples, the t test above proves that the difference between the means of all variables is statistically insignificant.<br />
<br />
The final training and testing datasets can be accessed through following links:<br />
[Train Data](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-2-cobra-kai/blob/main/data/Collegetraindata.csv)<br />
[Test Data](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-2-cobra-kai/blob/main/data/Collegetestdata.csv)

## **Build the Models**

We trained different machine learning models to predict the graduation rate of a college. The training and testing datasets for all the models are taken as described in the last section. The code for modelling can be found here: [Modelling](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-2-cobra-kai/blob/main/code/Models_Final.ipynb)

### **1. Multiple Regression:**

The target variable for our prediction model is a continuous variable. Since the goal of developing a machine learning model is to come up with the simplest model with the best possible performance, we first chose to train a multiple regression model on our dataset.

But before training the model, we need to make sure that the assumptions of regression are satisfied. The code for testing assumptions can be found here: [Regression Assumptions](https://github.com/msis5223-pds2-2022spring/msis-5223-deliverable-2-cobra-kai/blob/main/code/Regression_assumptions.ipynb)

**Linearity:** To check the linearity, we created scatterplots between independent variables and the dependent variable.

Most of the variables had an almost linear relationship with the graduation rate. We can see that through the following graphs.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/21.png)
![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/22.png)
![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/23.png)
![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/24.png)

Some of the variables like &#39;no\_of\_enrolled&#39;, &#39;no\_of\_FT&#39;, &#39;pct\_phd\_faculty&#39; etc. had a non-linear relationship with the &#39;grad\_rate&#39;.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/25.png)
![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/26.png)
![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/27.png)


If these variables do not satisfy other assumptions as well, we will drop them from our model.

**Correlation:**

Correlation between all the variables was calculated using .corr() function in Python. Following are the correlation values:

|                        | no\_of\_enrolled | outstate\_tuition | no\_of\_FT | no\_of\_PT | PctTop10HS | PctTop25HS | pct\_phd\_faculty | pct\_termianl\_faculty | pct\_alumni\_donors | sf\_ratio | total\_misc\_cost | accept\_rate | grad\_rate\_round |
| ---------------------- | ---------------- | ----------------- | ---------- | ---------- | ---------- | ---------- | ----------------- | ---------------------- | ------------------- | --------- | ----------------- | ------------ | ----------------- |
| no\_of\_enrolled       | 1                | \-0.15717         | 0.964652   | 0.513191   | 0.189524   | 0.232154   | 0.340429          | 0.314369               | \-0.18372           | 0.240368  | 0.14065           | \-0.15873    | \-0.01802         |
| outstate\_tuition      | \-0.15717        | 1                 | \-0.21787  | \-0.25155  | 0.553843   | 0.480738   | 0.374949          | 0.397766               | 0.559586            | \-0.54904 | 0.416994          | \-0.21667    | 0.585124          |
| no\_of\_FT             | 0.964652         | \-0.21787         | 1          | 0.570537   | 0.149561   | 0.20548    | 0.326267          | 0.305416               | \-0.23245           | 0.282792  | 0.134823          | \-0.16394    | \-0.07366         |
| no\_of\_PT             | 0.513191         | \-0.25155         | 0.570537   | 1          | \-0.09877  | \-0.04747  | 0.158701          | 0.150673               | \-0.28179           | 0.233253  | 0.145184          | \-0.10607    | \-0.25864         |
| PctTop10HS             | 0.189524         | 0.553843          | 0.149561   | \-0.09877  | 1          | 0.891046   | 0.526114          | 0.486867               | 0.437856            | \-0.37815 | 0.292571          | \-0.45083    | 0.499327          |
| PctTop25HS             | 0.232154         | 0.480738          | 0.20548    | \-0.04747  | 0.891046   | 1          | 0.541333          | 0.521563               | 0.405437            | \-0.28704 | 0.266418          | \-0.41458    | 0.480471          |
| pct\_phd\_faculty      | 0.340429         | 0.374949          | 0.326267   | 0.158701   | 0.526114   | 0.541333   | 1                 | 0.847218               | 0.242707            | \-0.12232 | 0.28943           | \-0.30873    | 0.330357          |
| pct\_termianl\_faculty | 0.314369         | 0.397766          | 0.305416   | 0.150673   | 0.486867   | 0.521563   | 0.847218          | 1                      | 0.262073            | \-0.15369 | 0.324143          | \-0.29338    | 0.309817          |
| pct\_alumni\_donors    | \-0.18372        | 0.559586          | \-0.23245  | \-0.28179  | 0.437856   | 0.405437   | 0.242707          | 0.262073               | 1                   | \-0.39426 | 0.064783          | \-0.10055    | 0.494582          |
| sf\_ratio              | 0.240368         | \-0.54904         | 0.282792   | 0.233253   | \-0.37815  | \-0.28704  | \-0.12232         | \-0.15369              | \-0.39426           | 1         | \-0.25609         | 0.097926     | \-0.3194          |
| total\_misc\_cost      | 0.14065          | 0.416994          | 0.134823   | 0.145184   | 0.292571   | 0.266418   | 0.28943           | 0.324143               | 0.064783            | \-0.25609 | 1                 | \-0.28022    | 0.230433          |
| accept\_rate           | \-0.15873        | \-0.21667         | \-0.16394  | \-0.10607  | \-0.45083  | \-0.41458  | \-0.30873         | \-0.29338              | \-0.10055           | 0.097926  | \-0.28022         | 1            | \-0.26433         |
| grad\_rate\_round      | \-0.01802        | 0.585124          | \-0.07366  | \-0.25864  | 0.499327   | 0.480471   | 0.330357          | 0.309817               | 0.494582            | \-0.3194  | 0.230433          | \-0.26433    | 1                 |

We can see that there is a high correlation between &#39;no\_of\_enrolled&#39; &amp; &#39;no\_of\_FT&#39; , &#39;PctTop25HS&#39; &amp; &#39;PctTop10HS&#39; and &#39;pct\_phd\_faculty&#39; &amp; &#39;pct\_termianl\_faculty&#39;. Therefore, we will remove one variable from each set so that our model is free from any bias.

**Homoscedasticity**

Next assumption we need to check is the constant variance. Following is the plot for Residual Values VS Predicted Values:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/28.png)

Since the plot is equally distributed on both sides of the X-axis, we can say that the assumption for homoscedasticity is satisfied.

**Normally Distributed Residuals**

To check normally distributed residuals, we created QQ Plot.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/29.png)

Looking at the plot above, we can say that the residuals are almost normally distributed. The tails suggest that the plot deviates a little from normality and is leptokurtic.

**Independence**

Durbin\_Watson test was conducted on the dataset to test for independence. The result came out to be non-significant; therefore, we can say that there is no evidence to suggest this assumption is violated.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/30.png)

We used sklearn library in Python to train and test the model. Firstly, the variables used for training were selected as per our domain knowledge. Then we dropped some variables based on the correlation assumption. Since the model was built on standardized variables, the regression coefficients were used to find the variable importance. Below is the bar graph that depicts the variables and their importance in predicting the Graduation Rate:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/31.png)

We can see that the variables – &#39;no\_of\_PT&#39;, &#39;outstate\_tuition&#39; and &#39;no\_of\_enrolled&#39; are the most important variables in our model. To check the performance of the model, the following metrics were calculated:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/42.png)

To visually understand the efficiency of model, we plotted a graph between the actual and the predicted graduation rate:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/32.png)

The scatter plot almost follows the straight line that suggests that our model can predict the dependent variable better than the base model.

**2. Random Forest Regressor**

A random forest regressor was built next to explore its predictive ability on Graduation Rate. Since random forests use bootstrapping to offset class imbalances, it was chosen to resolve any imbalance that might have been present in the dataset. A function was created to execute the model and visualize relevant metrics.

Additionally, the tree was iteratively run for 5 times to compare variability in predictions. The metrics of the 5 iterations are as shown below. We can see an overall stability in the predictions, accuracy, RMSE, and Rsquare, indicating that the model performance is robust.

All the 5 trees created predicted outstate\_tuition to be the most significant variable, followed by pct\_alumni\_donors, accept\_Rate, Percentage of top 25 high school students. The last significant variable appeared to be differently predicted during the first iteration. Hence we will compare the metrics of the models to decide the prediction of the best iteration.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/33.png)

Comparing RMSE, Rsquare and Accuracy of each iteration we find the 3rd iteration to have the best-fitting values in each metric. Looking at the results of the 3rd iteration (shown below), we find the last significant variable to be no of enrolled students.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/34.png)

Looking at the variables selected, clearly outstate tuition makes sense as the most important variable, as cost of tuition is a major concern that leads to students dropping out due to the pressure of debt or unaffordability. The other variable describe the value of education of a university, the acceptance rate, No of enrolled students, percentage of alumni donors and percentage of top 25 high school students – these are factors that add value to a university and talk about the caliber of it&#39;s students. These variables would play a significant role in affecting graduation rate as they could cause students to drop out when there is are changes in the university that affects these factors.

In conclusion, the random forest regressor determines the cost of education and factors pertaining to the quality of enrolled students to be the most significant variables in affecting graduation rate.

**3. Regression Tree:**

Next, a regression tree, which is a decision tree used to predict continuous target variables (Graduation Rate) instead of a discrete target variable. Apart from the managerial variable selection that we have done earlier, we will perform feature selection using ExtraTreesRegressor, that is, a model-based feature selection. Models such as regression trees are highly sensitive to the scale of the features and therefore, the features are normalized to ensure that they are all scaled accordingly.

The output of the extraTreesRegressor feature selection is as displayed below. Among the selected variables, &#39;outstate\_tuition&#39;, &#39;pct\_alumni\_donors&#39;, and top 10 and 25 high school students appear to be the most important variables in predicting the graduation rate. This implies that the graduation rate is associated with the financial aspect of the education and the caliber of a student&#39;s performance. We will further train our model using these variables to predict graduation rate.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/35.png)

First, a regression tree model is trained without specific hyperparameters – the metrics of the model is abysmal. With an R-square value of -0.087, the model performs worse than the baseline model. A scatterplot is displayed to visualize the expected and predicted values for the test data – there is no apparent pattern in the predicted and expected graduation rates. This model needs to be fine-tuned to make better predictions.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/36.png)


In order to find the best parameters for our RegressorTree model, we will utilize GridSearch Hyperparameter Tuning. Below are the parameters initialized for hyperparameter selection:


```
# Hyper parameters range intialization for tuning 

from sklearn.model\_selection import GridSearchCV
parameters={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }

```
The best parameters for the decision tree regressor based on the GridSearch evaluation is used in the RegressorTree. The results from the tuned model are as follows:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/43.png)

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/37.png)

The above plot is the visualization of predicted and expected graduation rate after tuning the model. The results are much better than the base model. Below is a visualization of the decision tree splits. This model considers percentage of alumni donors as the most important factor and uses it to split the dataset followed by student&#39;s highschool performance and outstate tuition fee.

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/38.png)

The R-square value has significantly improved from -0.08 to 0.38. This result suggests that the about 38% of variability in graduation rate can be predicted from the selected predictor variables (Outstate Tuition, Percentage of alum donors, Percentage of top 25 high school students, Percentage of top 10 high school students and acceptance rate of the university). In conclusion, the regressor model suggests that the financial cost of education in a university, and the caliber of the enrolled students play an important role in the graduation rate of the university.

**4. Neural Network**

Finally, we developed an Artificial Neural Network model to predict the graduation rate. This was done using Python library Keras. The input and the first hidden layer was added using &#39;relu&#39; as the activation function while the second layer was added with &#39;tanh&#39; as the activation function. &#39;adam&#39; was used as the optimizer algorithm that modified the attributes of the neural network like weights and learning rate. Also, the loss function was given as mean-squared error.

```
# create ANN model
model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=5,input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
 
# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
 
# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))
 
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')
```

Firstly, the model was fitted with a batch size of 20 and the number of epochs was given as 10. The following were the loss values at each epoch:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/39.png)

We could see that the loss function value decreased at every iteration. To find out the best parameters, we calculated the accuracy of several models using different permutations of batch size and epochs.

```
# Defining a function to find the best parameters for ANN
def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    
    # Defining the list of hyper parameters to try
    batch_size_list=[5, 10, 15, 20]
    epoch_list  =   [5, 10, 50, 100]
    
    import pandas as pd
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    # initializing the trials
    TrialNumber=0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber+=1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
 
            # Defining the Second layer of the model
            model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
 
            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))
 
            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')
 
            # Fitting the ANN to the Training set
            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)
 
            MAPE = np.mean(100 * (np.abs(y_test-model.predict(X_test))/y_test))
            
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)
            
            SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
                                                                    columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    return(SearchResultsData)
 
 
 
# Calling the function
ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)


```

**Output:** 

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/44.png)

The highest accuracy was found when the batch size was taken as 15 and epochs were 5. So, we trained the model and predicted the graduation rate using the above parameters. Once the predicted values were calculated, we then found the values for metrics to assess the model performance. Following were its results:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/45.png)

Model 2: To check if we could improve the performance of the neural network model, we changed the activation functions to &#39;tanh&#39; and &#39;elu&#39; in the first and second layers respectively.

```
# Defining the first layer of the model
model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='tanh'))

# Defining the Second layer of the model
model.add(Dense(units=5, kernel_initializer='normal', activation='elu'))

# The output neuron is a single fully connected node 
# Since we will be predicting a single number
model.add(Dense(1, kernel_initializer='normal'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')
```
We then followed the same approach as the last NN model for finding the parameters that give the highest accuracy. The best parameter came out to be batch\_size=15 and epochs=5

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/40.png)

Using these parameters, the dependent variables was predicted and the values for different metrics were calculated as follows:

![alt text](https://github.com/ssrivas1/SpringProject/tree/main/assets/41.png)

## **Model Assessment**

### **Strengths and Weaknesses of Models**

- **Multiple Regression**

While multiple regression models are fairly robust, their drawback is their tendency to have assumptions about the data. Multiple Regression mainly assumes its independent variables to have linear relationships with the target variable and this might not always be the case. But conversely, when its assumptions are satisfied, it is a solid and reliable predictor.

- **Artificial Neural Networks**

Artificial Neural Networks perform admirably in understanding complex relationships between dependent and independent variables. They don&#39;t require any assumptions to be satisfied or missing values to be handled. A major disadvantage however is their tendency to overfit the data or over learn the relationships in the training data, hence performing poorly when new data is presented.

- **Random Forest**

The use of bootstrapping by random forests is crucial when class imbalances are present in the data. Bootstrapping eliminates this problem by creating additional duplicate samples to pad up and offset the class imbalance. A drawback is it's ack of interpretability as there is no visual representation of the tree, since the predictions are made as an aggregated average of all the trees created.

- **Regression Tree**

Regression Trees have the advantage of being easily interpretable, with decisions and predictions neatly made by following the nodes of the trees. But a significant disadvantage it poses is its tendency to overfit the data, and hence performing poorly on new predictions.

### **Evaluation Metrics**

The following metrics were chosen for model selection -

1. Root Mean Squared Error &amp; Mean Absolute Error – These two metrics measure the model&#39;s accuracy in making predictions and hence represent its predictive power
2. R-Square - This metric was chosen as it represents the models ability to understand the relationship between the target and independent variables

The table below outlines the metrics of each model

| **Model** | **RMSE** | **R-Square** | **MAE** |
| --- | --- | --- | --- |
| Multiple Regression | 0.128 | 50.5% | 0.10 |
| Random Forest | 0.129 | 49.31% | 0.10 |
| Regression Tree | 0.148 | 38.57% | 0.119 |
| Neural Network 1 | 0.147 | 34.64% | 0.120 |
| Neural Network 2 | 0.141 | 39.54% | 0.115 |

#### **Selected Model:**

Looking at the results above, we find that all models except the Regression Tree perform quite similarly across all metrics. There are very minor differences in the results of the other 4 models.

The multiple regression model showcases the best results and is closely followed by the Random Forest Regressor. Though the multiple regression model has assumptions that need to be satisfied, various solutions exist to conduct transformations needed to execute the model. The random forest however suffers for a lack of interpretability and would be computationally expensive for a client to use if they have a significantly larger data set, while the Multiple Regression would be simple and robust. In addition to better performance Multiple Regression also clearly provides which variables would strongly affect the graduation rate.

In conclusion, due to the factors outlined above the multiple regression model was chosen as the best model in predicting graduation rate.

## **Conclusion and Discussion**

We chose Multiple regression as our final model based on its performance and interpretability. According to the multiple regression model, outstate tuition is the most important variable affecting graduation rates. As mentioned above this seems to be logically coherent as higher outstate tuition would lead to higher dropout rates for students who fail to finance their education.

Hence to improve graduation rates, colleges could introduce scholarship programs for financially underprivileged students. In addition, colleges could partner up with financing companies to provide better education loan rates for their students which would not only lead to an academically stronger cohort but would also help increase social mobility for such students.

It must be noted that there could be further improvements made to our findings by enriching the data by including more variables. More data could also be collected from varied colleges to help reduce bias in our data. Improving the data and the models is a subject of further research
