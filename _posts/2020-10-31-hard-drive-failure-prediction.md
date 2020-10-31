<center>
    <h1>Hard Drive Failure Prediction</h1>
</center>

<p align="center">
    <img src="https://user-images.githubusercontent.com/54760460/94100169-8162eb00-fe4a-11ea-93eb-b5972127ca1d.png" alt="Hard Drive Failure"/><br />
    <a href="https://vsbytes.com/">Image Source</a>
</p>



## Contents
1. Introduction to the Business Problem
2. Data Extraction
3. Existing Approaches
4. Improvements to the Existing Approaches
5. Evaluation Metrices
6. Exploratory Data Analysis
7. Data Preprocessing
8. Feature Engineering
9. Modelling
10. Comparision of Different Models
11. Future Work
12. References

## 1. Introduction to the Business Problem

- Hard drives are essential parts of data storage. When a hard disk drive malfunctions and data can’t be accessed, it is called a Hard Disk Drive Failure.
- There are many reasons for the failure of a Hard Drive like high magnetic fields, exposure to heat, or any normal operation, data corruption, human error, power issues etc,.
- Failure of a hard drive can be immediate, progressive or limited. Data may be totally destroyed or partially destroyed or can be totally recovered.
- So predicting the failures can be helpful in data backup prior to failure and we can replace the drive with a good one without any loss of our data.
- A Hard drive failure prediction method called SMART(Self-Monitoring, Analysis and Reporting Technology) has been proposed to constantly monitor the drives to predict failures in order to reduce the risk of data loss.
- SMART attributes represent Hard Drive health statistics such as the number of scan errors, reallocation counts and probational counts of a Hard Drive.
- In this case study, we are using Backblaze Hard Drive Stats Q3 2019 to predict failures of Hard Drives.
- Backblaze provides the data containing information from different manufacturers and different models with all the SMART parameters.
- Using the data provided by Backblaze, we apply different machine learning algorithms to predict the failures.

To pose the problem in a better way, we can say that we need to predict if a hard drive is going to fail in the next 'N' days('N' is optimal value). If we can predict failure before 'N' days, we get sufficient time to retrieve the data and can replace that with a new drive.

## 2. Data Extraction

The Backblaze data center takes a snapshot of each operational hard drive everyday. The snapshot includes basic drive information along with the SMART statistics reported by the drive. All drives' snapshots for a given day are collected into a file consisting of a row for each active hard drive. The format of this file is 'csv'(Comma Separated Values). Each day this file is named in the format YYYY-MM-DD.csv, for example, 2019-08-01.csv.

The columns of the file are as follows:

- Date – The date of the file in yyyy-mm-dd format. 
- Serial Number – The manufacturer-assigned serial number of the drive. 
- Model – The manufacturer-assigned model number of the drive. 
- Capacity – The drive capacity in bytes. 
- Failure – Contains a '0' if the drive is OK. Contains a '1' if this is the last day the drive was operational before failing.
- SMART Parameters

We can download data from Backblaze website:
[https://www.backblaze.com/b2/hard-drive-test-data.html#downloading-the-raw-hard-drive-test-data](https://www.backblaze.com/b2/hard-drive-test-data.html#downloading-the-raw-hard-drive-test-data)

## 3. Existing Approaches

Research Paper: [https://hal.archives-ouvertes.fr/hal-01703140/document](https://hal.archives-ouvertes.fr/hal-01703140/document)

- In this research paper, machine learning algorithms were applied on the 2013 Backblaze dataset and all the models’ performances were compared. 
- 12 million samples from 47,793 drives including 31 models from 5 manufacturers were used. Of the 12 million samples, only 2586 samples have failure labels set to 1 and others are healthy samples which makes the dataset highly imbalanced. 
- Defined a time window for failure which predicts whether the hard drive is going to fail in the next N days. 
- Some SMART parameters were pre selected based on high correlation to failure events. SMART parameters - 5, 12, 187, 188, 189, 190, 198, 199, 200. 
- To handle data imbalance, SMOTE technique was applied. 
- Some failure samples share the exact same feature values as healthy samples leading to the impossibility to discriminate against them and so certain categories of failure samples were filtered.
- Logistic Regression, Support Vector Machine, Random Forest Classifier, Gradient Boosting Decision Trees - All these machine
learning algorithms were applied on the final data. 
- It was observed that RF and GBDT resulted in good precision and recall. 
- RF provided best performances with 95% precision and 67% recall.

## 4. Improvements to the Exisiting Approaches

- For every hard drive, SMART parameters are recorded everyday and these parameters are very important in predicting failures.
- We can extract time series features from these SMART parameters like rolling window features(mean, std), expanding window features(mean, std), exponential smoothing, lags, etc,.
- In the research paper, it was given that SMOTE technique used for data balancing didn't contributed much in predicting failures. Hence upsampling is used to balance the data.

## 5. Evaluation Metrics

False alarm rate is a good choice for balanced datasets but, as operational datasets are extremely unbalanced in favour of working drives, even a low false alarm rate in the range of 1% could translate into poor performances. Therefore, we report precision, recall metrics and f1-score.

False Alarm Ratio = False Alarms/Total number of alarms<br />
i.e., False Alarm Ratio = (Number of drives wrongly detected as failures)/(Total number of actually failed drives)

Precision = true positive/(true positive + false positive)<br />
i.e., Precision = (Number of drives that are actually failures)/(Total number of drives that are predicted to be failures)

Recall = true positive/(true positive + false negative)<br />
i.e., Recall = (Number of drives predicted correctly as failures)/(Total number of drives that are actually failures)

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## 6. Exploratory Data Analysis

Backblaze uses the following five SMART stats as a means to help determine if a drive is going to fail.

    ATTRIBUTE       DESCRIPTION
    SMART 5         Reallocated Sectors Count
    SMART 187       Reported Uncorrectable Errors
    SMART 188       Command Timeout
    SMART 197       Current Pending Sector Count
    SMART 198       Uncorrectable Sector Count

Along with above features, we also selected other SMART features - SMART 5, 9, 12, 187, 188, 189, 190, 193, 194, 197, 198, 199, 200, 241, 242.

    df=pd.read_csv("july_august"+"\\"+listdir("july_august")[0])
    for file in listdir("july_august")[1:]:
      df=df.append(pd.read_csv("july_august"+"\\"+file))

Output:
![image](https://user-images.githubusercontent.com/54760460/94002334-64310c80-fdb7-11ea-9b9b-3b8b47492618.png)

As we can observe, there are normalized features for SMART parameters.<br />
But, we only select raw values. Also, due to limited computational power, we select only segate models' July and August months' data.

    features=['date','model','serial_number','capacity_bytes', 'failure','smart_5_raw','smart_9_raw','smart_12_raw','smart_187_raw',
             'smart_188_raw','smart_189_raw','smart_190_raw','smart_193_raw','smart_194_raw','smart_197_raw','smart_198_raw',
             'smart_199_raw','smart_200_raw','smart_241_raw','smart_242_raw']
    df_new=df[features]
    df_segate=df_new[df_new['model']=='ST4000DM000']
    for model in df_new['model'].unique()[1:]:
        if model[:2]  == 'ST' or model[:2] == 'Se':
            print(model)
            row_df=df_new[df_new['model']==model]
            df_segate=pd.concat([df_segate,row_df])
    df_segate.reset_index(inplace=True,drop=True)
  
### 6.1. Univariate Analysis: Failure

    sb.set_style('darkgrid')
    sb.distplot(df_segate['failure'])
    plt.title("Distribution Plot of failure")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94002994-444e1880-fdb8-11ea-9c13-5952309bb68d.png)

    df_segate[df_segate['failure']==0].shape
    (5053997, 20)
    
    df_segate[df_segate['failure']==1].shape
    (367, 20)
 
 **From the above plot, we can observe that the data set is highly imbalanced with only a few number of failures(failure=1)**
 
### 6.2. Univariate Analysis: smart_5_raw
 
    sb.boxplot(x='failure',y='smart_5_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_5_raw")
    plt.show()
  
![image](https://user-images.githubusercontent.com/54760460/94003510-ee2da500-fdb8-11ea-8c8e-592397c561a1.png)

**From the above plot, we can observe that most of the drives are working for different range values of smart_5_raw and most of failures occurred when smart_5_raw is below the range of 10000**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_5_raw'].dropna())
    plt.title("Distribution Plot of smart_5_raw for failed drives")
    plt.show()

![image](https://user-images.githubusercontent.com/54760460/94003648-27661500-fdb9-11ea-802a-d66ec4b02aa3.png)

**From the above plots, we can observe that most of the failures(failure=1) occurred when smart_5_raw values are in the range of 0 to 10000 and few failures in the range of 20000 and one failure in the range of 70000**

### 6.3. Univariate Analysis: smart_9_raw

    sb.boxplot(x='failure',y='smart_9_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_9_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94004052-b70bc380-fdb9-11ea-8d15-71dba6647e63.png)

**From the above plot, we can observe that working drives have smart_9_raw values in the range of 12000 to 29000. However, some failed drives also have smart_9_raw values in the range of 10000 to 18000**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_9_raw'].dropna())
    plt.title("Distribution Plot of smart_9_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94004148-de629080-fdb9-11ea-91ae-64c486340776.png)

**From the above plots, we can observe that most of the failures occurred when smart_9_raw is in the range of 10000 to 20000**

### 6.4. Univariate Analysis: smart_187_raw

    sb.boxplot(x='failure',y='smart_187_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_187_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94004304-1bc71e00-fdba-11ea-8387-1dea99988bc0.png)

**From the above plot, we can observe that working drives have smart_187_raw values in different ranges and most of the failures when smart_187_raw value is 0**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_187_raw'].dropna())
    plt.title("Distribution Plot of smart_187_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94004378-3ac5b000-fdba-11ea-9a45-3a93447f05c5.png)

**From the above plots, we can observe that most of the failures occurred when smart_187_raw values are in the range of 0 to 1000 and only one failure occurred when the values are greater than 50000**

### 6.5. Univariate Analysis: smart_188_raw

    sb.boxplot(x='failure',y='smart_188_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_188_raw")
    plt.show()
 
![image](https://user-images.githubusercontent.com/54760460/94006322-4a92c380-fdbd-11ea-9812-dc098e882a95.png)

**From the above plot, we can observe that working drives have smart_188_raw values in very high ranges of 1e11**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_188_raw'].dropna())
    plt.title("Distribution Plot of smart_188_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94006406-68602880-fdbd-11ea-8b28-e879667ab7c5.png)

**From the above plots, we can observe that most of the failures are when smart_188_raw value is zero and few failures occured when the values are very high**

### 6.6. Univariate Analysis: smart_193_raw

    sb.boxplot(x='failure',y='smart_193_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_193_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94007403-086a8180-fdbf-11ea-88b6-3bdf69391419.png)

**From the above plot, we can observe that working drives have smart_193_raw values in different ranges from 0 to 1400000**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_193_raw'].dropna())
    plt.title("Distribution Plot of smart_193_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94007467-21733280-fdbf-11ea-8ae6-326711e201cb.png)

**From the above plots and analysis, we can observe that most of the failures occurred when smart_193_raw is in the range of 0 to 10000 and few failures in the range of 10000 to 50000**

### 6.7. Univariate Analysis: smart_194_raw

    sb.boxplot(x='failure',y='smart_194_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_194_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94008381-909d5680-fdc0-11ea-88c4-a7aacac79dfa.png)

**From the above plot, we can observe that both working and failed drives have smart_194_raw values in almost same ranges from 25 to 35**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_194_raw'].dropna())
    plt.title("Distribution Plot of smart_194_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94008471-b4f93300-fdc0-11ea-9b76-56dfa2f0ace5.png)

**From the above plots, we can observe that most failures occurred when smart_194_raw values are in the range of 20 to 40**

### 6.8. Univariate Analysis: smart_197_raw

    sb.boxplot(x='failure',y='smart_197_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_197_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94008770-29cc6d00-fdc1-11ea-9359-0271ab8cfb9c.png)

**From the above plot, we can observe that working drives have smart_197_raw values in the range of 0 to 3000 and also failed drives have same range of values**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_197_raw'].dropna())
    plt.title("Distribution Plot of smart_197_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94008922-5aaca200-fdc1-11ea-80fc-355a541b01cd.png)

**From the above plots, we can observe that all failures occurred when smart_197_raw values are in the range of 0 to 3000 and most failures occurred when the value is 0**

### 6.9. Univariate Analysis: smart_198_raw

    sb.boxplot(x='failure',y='smart_198_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_198_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94009045-83cd3280-fdc1-11ea-8d37-b8a6b504425b.png)

**From the above plot, we can observe that working drives have smart_198_raw values in the range of 0 to 3000 and failed drives also have almost same range of values**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_198_raw'].dropna())
    plt.title("Distribution Plot of smart_198_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94009177-bd05a280-fdc1-11ea-974f-692287c30bd8.png)

**From above plot, we can observe that all the failures occurred when smart_198_raw values are in the range of 0 to 2500 and most failures occurred when the value is 0**

### 6.10. Univariate Analysis: smart_241_raw

    sb.boxplot(x='failure',y='smart_241_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_241_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94009469-21286680-fdc2-11ea-930d-99bb6c0d8ccb.png)

**From the above plot, we can observe that working drives have smart_241_raw values in different ranges and have high values in the range of 1e11**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_241_raw'].dropna())
    plt.title("Distribution Plot of smart_241_raw for failed drives")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94009668-6b114c80-fdc2-11ea-834d-d3ac1855b57d.png)

**From the above plot, we can observe that failures occurred when smart_241_raw values are in very high ranges of 0.5e11 to 0.6e11**

### 6.11. Univariate Analysis: smart_242_raw

    sb.boxplot(x='failure',y='smart_242_raw',data=df_segate)
    plt.title("Box Plot to detect failure based on smart_242_raw")
    plt.show()
    
![image](https://user-images.githubusercontent.com/54760460/94010455-96486b80-fdc3-11ea-9861-2230de7e32f1.png)

**From the above plot, we can observe that working drives have smart_242_raw values in different ranges of values from 0 to 2.5e13**

    sb.distplot(df_segate[df_segate['failure']==1]['smart_242_raw'].dropna())
    plt.title("Distribution Plot of smart_242_raw for failed drives")
    plt.show()
  
![image](https://user-images.githubusercontent.com/54760460/94010707-ed4e4080-fdc3-11ea-90c9-3c2ca265983b.png)

**From the above plot and analysis, we can observe that most of the failures occurred when smart_242_raw values are in the range of 1.062508e+10 to 1.5e+11**

## Summary of Univariate Analyses:

From all the above plots, we can observe that the failures occurred for different ranges of smart attributes' values.<br />
Most failures occurred when smart attributes' values are 0.

## 7. Data Preprocessing

### 7.1. Missing Values

We remove features which have missing values' percentage greater than 30.

    '''percentage of mssing values in each column'''
    features_new_2=[]
    for i in df_segate.columns:
        print(i,null_sum[i]*100/df_segate.shape[0])
        if null_sum[i]*100/df_segate.shape[0] > 30:
            continue
        features_new_2.append(i)
    df_segate=df_segate[features_new_2
        
**More than 25% of the values are missing in smart_189_raw, smart_200_raw. So dropping those columns**

#### 7.1.1. Filling Missing Values with mean

We fill missing values in each column with their mean

    df_segate['smart_5_raw']=df_segate['smart_5_raw'].fillna(df_segate['smart_5_raw'].mean())
    df_segate['smart_9_raw']=df_segate['smart_9_raw'].fillna(df_segate['smart_9_raw'].mean())
    df_segate['smart_12_raw']=df_segate['smart_12_raw'].fillna(df_segate['smart_12_raw'].mean())
    df_segate['smart_187_raw']=df_segate['smart_187_raw'].fillna(df_segate['smart_187_raw'].mean())
    df_segate['smart_188_raw']=df_segate['smart_188_raw'].fillna(df_segate['smart_188_raw'].mean())
    df_segate['smart_190_raw']=df_segate['smart_190_raw'].fillna(df_segate['smart_190_raw'].mean())
    df_segate['smart_193_raw']=df_segate['smart_193_raw'].fillna(df_segate['smart_193_raw'].mean())
    df_segate['smart_194_raw']=df_segate['smart_194_raw'].fillna(df_segate['smart_194_raw'].mean())
    df_segate['smart_197_raw']=df_segate['smart_197_raw'].fillna(df_segate['smart_197_raw'].mean())
    df_segate['smart_198_raw']=df_segate['smart_198_raw'].fillna(df_segate['smart_198_raw'].mean())
    df_segate['smart_199_raw']=df_segate['smart_199_raw'].fillna(df_segate['smart_199_raw'].mean())
    df_segate['smart_241_raw']=df_segate['smart_241_raw'].fillna(df_segate['smart_241_raw'].mean())
    df_segate['smart_242_raw']=df_segate['smart_242_raw'].fillna(df_segate['smart_242_raw'].mean())

### 7.2. Computing mean, std, min, max for smart_parameters row-wise

We calculate Mean, Standard Deviation, Min, Max for SMART parameters row-wise and add them as new features.

    df_segate['mean']=df_segate[df_segate.columns[5:18]].mean(axis=1)
    df_segate['std']=df_segate[df_segate.columns[5:18]].std(axis=1)
    df_segate['min']=df_segate[df_segate.columns[5:18]].min(axis=1)
    df_segate['max']=df_segate[df_segate.columns[5:18]].max(axis=1)

### 7.3. Removing negative capacity byte values

We have observed that there are some negative values for capacity byte which are wrongly recorded. Hence dropping those rows.

    temp=df_segate[df_segate['capacity_bytes']<10.0]
    ind=temp.index
    df_segate.drop(df_segate.index[list(ind)],inplace=True)

## 8. Feature Engineering

Sort the DataFrame by serial_number and date to extract time series features.

    df_new=pd.read_csv('df_segate_july_august_no_backtrack.csv')
    df_new_with_lag=df_new.sort_values(['serial_number','date'])

### 8.1. Rolling Mean, Standard Deviation for SMART parameters - window 15

We can write code using pd.DataFrame.shift function but we shouldn't do that directly for a column.<br /> 
We have different models with different serial numbers.<br /> 
For every unique serial number, we need to calculate rolling mean and standard deviation.<br />
We can write code with shift function by looping over all the unique serial numbers but it takes a lot of time as there are many thousands of unique serial numbers.<br />
Hence written code as pointed below.<br />
Here we take window=15.

    serial_numbers=df_new_with_lag['serial_number'].values
    serial_number=df_new_with_lag['serial_number'].values[0]
    for column in tqdm(df_new_with_lag.columns[5:18]):
        rolling_mean=[]
        rolling_stdev=[]
        for i in range(df_new_with_lag.shape[0]):
            if serial_numbers[i]!=serial_numbers[i-1]:
                values=[] 
                values.append(df_new_with_lag[column].values[i])
                rolling_mean.append(mean(values))
                rolling_stdev.append(values[-1])
            else:
                if(len(values)<15): 
                    values.append(df_new_with_lag[column].values[i])
                    mean_=mean(values[0:len(values)])
                    stdev_=stdev(values[0:len(values)])
                    rolling_mean.append(mean_)
                    rolling_stdev.append(stdev_)
                else:
                    values.append(df_new_with_lag[column].values[i])
                    mean_=mean(values[len(values)-15:len(values)])
                    stdev_=stdev(values[len(values)-15:len(values)])
                    rolling_mean.append(mean_)
                    rolling_stdev.append(stdev_)
        df_new_with_lag[column+'_rolling_mean'] = rolling_mean
        df_new_with_lag[column+'_rolling_stdev'] = rolling_stdev
        
### 8.2. Expanding Mean for SMART parameters

Below is the code snippet for extracting expanding mean features.

    serial_numbers=df_new_with_lag['serial_number'].values
    serial_number=df_new_with_lag['serial_number'].values[0]
    for column in tqdm(df_new_with_lag.columns[5:18]):
        expanding_mean=[]
        expanding_stdev=[]
        for i in range(df_new_with_lag.shape[0]):
            if serial_numbers[i]!=serial_numbers[i-1]:
                values=[] 
                values.append(df_new_with_lag[column].values[i])
                expanding_mean.append(sum(values))
                expanding_stdev.append(values[-1])
            else:
                values.append(df_new_with_lag[column].values[i])
                expanding_mean.append(mean(values))
                expanding_stdev.append(stdev(values))
        df_new_with_lag[column+'_expanding_mean'] = expanding_mean
        df_new_with_lag[column+'_expanding_stdev'] = expanding_stdev
        
### 8.3. Exponential Smoothing

For exponential smoothing, we have considered alpha=0.15.

    serial_numbers=df_new_with_lag['serial_number'].values
    serial_number=df_new_with_lag['serial_number'].values[0]
    alpha=0.15
    for column in tqdm(df_new_with_lag.columns[5:18]):
        predicted_values=[]
        for i in range(df_new_with_lag.shape[0]):
            if serial_numbers[i]!=serial_numbers[i-1]:
                predicted_value = (df_new_with_lag[column].values)[i]
                predicted_values.append(predicted_value)
            else:
                predicted_value =(alpha*df_new_with_lag[column].values[i]) + ((1-alpha)*predicted_value)
                predicted_values.append(predicted_value)
        df_new_with_lag[column+'_exp_avg'] = predicted_values
    df_new_with_lag=df_new_with_lag.sort_values(['date'])
    
### 8.4. Change Failure status by backtracking

We need to predict whether a drive is going to fail in the next 'N' days. Here we take N=15.<br />
For this, we backtrack last 15 days' failures, i.e., if we have a failed drive, we mark failure as '1' for its previous 15 days. 

    #backtrack last 15 days failures
    df_segate_backtrack=pd.read_csv("df_segate.csv")
    new_date=[]
    for date in df_segate_backtrack['date']:
        new_date.append(datetime.strptime(date,'%Y-%m-%d').date())
    df_segate_backtrack['date']=new_date

    failed=df_segate_backtrack[df_segate_backtrack['failure']==1]
    for serial_number in tqdm(failed.serial_number):
        d=failed[failed['serial_number']==serial_number].date.values
        temp=df_segate_backtrack[df_segate_backtrack['serial_number']==serial_number]
        temp=temp[temp.date>=(d[0]-timedelta(days=15))]
        temp=temp[temp.date<d[0]]
        indices=temp.index
        df_segate_backtrack.loc[indices,'failure']=1
    failed=df_segate_backtrack[df_segate_backtrack['failure']==1]
    
### 8.5. Train-Test split, Response Coding Categorical Features and Upsampling Minority Class

After extracting the time series features and backtracking the failure status, we have splitted the data into train, cv, and test.<br />
We have extracted features from Model ID and Serial Number and encoded them using response coding.<br />
Below are the features extracted from Model ID and Serial Number:
 - model_char_count - eg: Model ID = 'ST12000NM0007', model_char_count = 13
 - model_second_last_char - eg: Model ID = 'ST12000NM0007', model_second_last_char = 'T7'
 - serial_number_second_last_char - eg: Serial Number = 'ZJV2SP8Y', serial_number_second_last_char = 'JY'

By response coding model_second_last_char and serial_number_second_last_char, we get working and failing probabilites for these features out of which we used working probabilites as features.

Finally, to balance the data, we upsampled the minority class(drives with failure=1).

    failed_train_upsample_df=resample(failed_train_df,replace=True,
                                      n_samples=int(working_train_df.shape[0]),
                                      random_state=42)
                                      
## 9. Modelling

Data is standardized and used for Logistic Regression and Support Vector Machines.<br />
For Naive Bayes, data is normalized.<br />
Data is used directly for Random Forests and XGBoost as these are based on decision trees.<br />
For all the below models, small code snippets and results are printed.

### 9.1. Logistic Regression

Logistic Regression is one of the useful algorithms for Binary Classification. The assumption that is made for this algorithm is that the data is linear.[[1]](https://hal.archives-ouvertes.fr/hal-01703140/document)[[2]](http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf)<br />

Hyper-parameter tuning is performed for different c_range values and L2 Regularization is used.

    c_range=[10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4]
    for c in c_range:
        LR_model=LogisticRegression(C=c,penalty='l2')
        LR_model.fit(train_standardized,train_output)
 
Observed that f1 score is very less with logistic regression.<br />
For c=0.01 and with penalty='l2', we got the best test f1 score.
- train_f1_score=0.751285
- cv_f1_score=0.009825
- test_f1_score=0.009673

### 9.2. Support Vector Machine

Support Vector Machine (SVM) alogirthm relies on finding the hyperplane that splits the two classes to predict while maximizing the distance with the closest data points.[[1]](https://hal.archives-ouvertes.fr/hal-01703140/document)[[2]](http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf)<br />

SGDClassifier with loss='Hinge' is used and hyper-parameter tuning is performed for different ranges of alpha.

    alpha_range=[10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4]
    for a in alpha_range:
        SGD_model=SGDClassifier(loss='hinge',alpha=a)
        SGD_model.fit(train_standardized,train_output)

Observed that test f1 score is less with SVM.<br />
For alpha=0.0001, we got the best test f1 score.
- train_f1_score=0.756617
- cv_f1_score=0.011170
- test_f1_score=0.01123

### 9.3. Random Forest Classifier

Random forest is an ensemble technique based on Decision Tress. It takes a subset of observations and a subset of variables to build a group of decision trees. Predictions are made based on a vote among the different decision trees. Random forest model is chosen as it is robust to noise, caused by poorly correlated SMART features.[[1]](https://hal.archives-ouvertes.fr/hal-01703140/document)[[2]](http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf)<br />

Hyper-parameter tuning is performed for different values of n_estimators, max_depth, learning_rate.<br />
Probabilities are calibrated using CalibratedClassiferCV before predicting the target values.

    n_estimators = [100,150,200]
    max_depth = [7,  9]
    for i in n_estimators:
        for j in max_depth:
            rf_model = RandomForestClassifier(n_estimators=i, max_depth=j)
            rf_model.fit(train_df_final_1,train_output)
            cal_rf_model=CalibratedClassifierCV(rf_model,method='isotonic',cv='prefit')
            cal_rf_model.fit(cv_df_final_1,cv_output)

Observed that test f1 score is less with RandomForestClassifier.<br />
For n_estiamtors=150, max_depth=9, we got the best test f1 score.
- train_f1_score=0.27872
- cv_f1_score=0.19057
- test_f1_score=0.02651

### 9.4. XGBClassifier

Gradient boosted tree (GBT) is another ensemble technique based on decision trees. Training takes place in an iterative fashion with the goal of trying to minimize a loss function using a gradient descent method.[[1]](https://hal.archives-ouvertes.fr/hal-01703140/document)[[2]](http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf)<br />

Hyper-parameter tuning is performed for different values of n_estimators and max_depth.<br />
Probabilities are calibrated using CalibratedClassiferCV before predicting the target values.

    n_estimators = [100, 150, 200, 500, 1000]
    max_depth = [7, 9]
    for i in n_estimators:
        for j in max_depth:
            xgb_model = xgb.XGBClassifier(n_estimators=i, max_depth=j,tree_method='exact')
            xgb_model.fit(train_df_final_1,train_output)
            cal_xgb_model=CalibratedClassifierCV(xgb_model,method='isotonic',cv='prefit')
            cal_xgb_model.fit(cv_df_final_1,cv_output)

    important_features=xgb_model.get_booster().get_score(importance_type="gain")
    important_features_sorted=sorted(important_features.items(),key=lambda x:x[1], reverse=True)
    index_features=dict()
    i=0
    for column in train_df_final.columns:
        index_features['f'+str(i)]=column 
        i+=1
    sorted_important_features_dict=dict()
    for i in important_features_sorted:
        key=i[0]
        sorted_important_features_dict[key]=index_features[key]
    print("Top 10 important features:")
    list(sorted_important_features_dict.items())[0:10]

![image](https://user-images.githubusercontent.com/54760460/94026129-eb41ad00-fdd6-11ea-9174-dd92c352bc9e.png)

Observed that **XGBoost performed very well in predicting failed hard drives.**<br />
Optimal hyper-parameters: **n_estimators=1000, max_depth=9**
- **test f1_score: 0.929026**
- **test Precison : 0.943334**
- **test Recall : 0.915139**

### 9.5. Naive Bayes

The Naive Bayes classifier makes the assumption that the value of a feature is conditionally independent of the value of another feature given some class label. Among the different techniques used for building Naive Bayes models, we chose Multinomial Naive Bayes, which assumes that the probability of a feature value given some class
label is sampled from a multinomial distribution. For regularization, we use Laplace smoothing.[[2]](http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf)<br />

Hyper-parameter tuing is performed for different ranges of alpha.


    alpha_range=[10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4]
    for a in alpha_range:
        nb_clf=MultinomialNB(alpha=a, fit_prior=True)
        nb_clf.fit(train_normalized,train_output)

Observed that test f1 score is less with Naive Bayes.<br />
For alpha=0.0001, we got the best test f1 score.
- train_f1_score=0.625518
- cv_f1_score=0.004463
- test_f1_score=0.004925

### 9.6. XGBClassifier with top 50 important features

    n_estimators = [100, 150, 200, 500, 1000]
    max_depth = [7, 9]
    for i in n_estimators:
        for j in max_depth:
            xgb_model_imp = xgb.XGBClassifier(n_estimators=i, max_depth=j,tree_method='exact')
            xgb_model_imp.fit(train_df_final_imp_1,train_output)
            cal_xgb_model_imp=CalibratedClassifierCV(xgb_model_imp,method='isotonic',cv='prefit')
            cal_xgb_model_imp.fit(cv_df_final_imp_1,cv_output)
            
Observed that f1 score is good with XGBClassifier when compared with all other models.<br />
With n_estimators=1000, max_depth=9, we got the best f1 score.
- **test f1_score : 0.935266**
- **test Precison : 0.9370764762826719**
- **test Recall : 0.9334619093539055**

### 9.7. Ensemble of RandomForestClassifier and XGBClassifier with top 50 important features

Tried ensembling with different combinations of above classifiers with different weights using EnsembleVoteClassifier.<br />
Ensemble of RandomForestClassifer and XGBClassifier with more weight to XGBClassifier resulted in good prediction.

    ensemble_vote_clf_imp = EnsembleVoteClassifier(clfs=[cal_rf_model_imp,cal_xgb_model_imp_new], voting='soft',refit=False,weights=[0.1,0.9])
    ensemble_vote_clf_imp.fit(train_df_final_imp_1,train_output)
    predicted_test_failure=ensemble_vote_clf_imp.predict(test_df_final_imp_1)
    test_f1_scores.append(f1_score(test_output, predicted_test_failure))
    print("test Precison :",precision_score(test_output,predicted_test_failure))
    print("test Recall :",recall_score(test_output,predicted_test_failure))
    print("test_f1_score=",test_f1_scores[-1])
    
- test Precison : 0.9477317554240631
- test Recall : 0.926711668273867
- test_f1_score= 0.9371038517796197

## 10. Comparision of Different Models

![image](https://user-images.githubusercontent.com/54760460/94027919-feee1300-fdd8-11ea-84d8-4223c5c5e127.png)

![image](https://user-images.githubusercontent.com/54760460/94028017-1b8a4b00-fdd9-11ea-842a-8f9906409542.png)

By comparing results of above two tables, we can observe that with top 50 important features we are able to get good scores.<br /> 
We got best f1 score with XGBClassifier.<br />
The next good model is RandomForest but it doesn't perform that well.

![image](https://user-images.githubusercontent.com/54760460/94028555-8dfb2b00-fdd9-11ea-9081-34a91ec24801.png)

### Summary:

1. From the above tables, we can observe that precision is good with random forests and xgboost but is very less with other classifiers.
2. Recall is good with all classifiers except random forests.
3. XGBClassifier predicted failed hard drives very well.
4. Precision and recall scores are highest with XGBClassifier and also with ensemble of RF and XGB.

5. **XGB With top 50 important features:**

    Test Precison : 0.937076<br />
    **Test Recall : 0.933461 -- Best**<br />
    Test f1_score: 0.935266

6. **Ensemble of XGB and RF With top 50 important features:**

    **Test Precison : 0.947731 -- Best**<br />
    Test Recall : 0.926711<br />
    **Test f1_score: 0.937103 -- Best**

7. We can observe that recall score is high with XGB classifier when top 50 features are used. Whereas f1-score and precision are high with ensemble of XGB and RF with top 50 features. We can choose any of these two models. Both the models are good.
8. Extracted many time series features from given data like exponential averages, rolling mean, rolling standard deviation, expanding mean, expanding standard deviation, backtracked last 15 days' failures etc,.
9. Top 10 important features for XGBClassifier are:
    smart_188_raw_exp_avg
    smart_5_raw
    smart_197_raw
    model_second_last_char_working
    capacity_bytes
    smart_199_raw_expanding_stdev
    serial_second_last_char_working
    smart_188_raw_expanding_mean
    smart_9_raw_rolling_mean
    smart_12_raw_rolling_stdev
10. The above results are on SEGATE model hard drives' July and August months data. We can try with XGBoost modelling for other hard drives also.
11. Recall is the important metric here. Our main aim to detect failed hard drives. In this case study, we have predicted hard drives that are going to fail in the next 15 days. If we can predict the drives that are going to fail few days before the failure, we can have sufficient time to retrieve data and replace them with new hard drives. It is somewhat fine if a drive predicted to be a failure is actually a working one. But the important aim here is recall: drives which are actually failures should be predicted as failures else if wrongly predicted as working ones, it may fail in future and data can't be retrieved.
11. We got best recall score with XGBClassifier with top 50 important features: **0.933461**
12. Limited data to arorund 5 million due to limited system capacity. Train data is around 3 millions(after upsampling around 6 million) With more amounts of data and feature engineering, we can further improve recall and f1 scores.

## 11. Future Work

1. From the above summary, we can observe that many of the time series features are useful in predicting hard drive failures.
2. So, we can extract and experiment with more time series features for much better results.<br /> 
    Eg: double/triple exponential smoothing, lag features, etc,.
3. We can use more data from all quarters of an Year to improve the results.
4. We can also experiment with deep learning techniques to predict the failures.

## 12. References

1. [https://hal.archives-ouvertes.fr/hal-01703140/document](https://hal.archives-ouvertes.fr/hal-01703140/document)
2. [http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf](http://cs229.stanford.edu/proj2017/final-reports/5242080.pdf)
3. [https://www.kaggle.com/vishakg/predicting-hdd-failures-using-ml](https://www.kaggle.com/vishakg/predicting-hdd-failures-using-ml)
4. [https://en.wikipedia.org/wiki/Hard_disk_drive_failure](https://en.wikipedia.org/wiki/Hard_disk_drive_failure)
5. [https://www.backblaze.com/blog/backblaze-hard-drive-stats-q3-2019/](https://www.backblaze.com/blog/backblaze-hard-drive-stats-q3-2019/)
6. [https://neurospace.io/blog/2018/10/predicting-hard-drive-failure-with-machine-learning/](https://neurospace.io/blog/2018/10/predicting-hard-drive-failure-with-machine-learning/)
7. [https://vsbytes.com/hdd-vs-ssd/](https://vsbytes.com/hdd-vs-ssd/)

**Thanks for Reading😃**

**Complete code in github: [https://github.com/VyshnaviVanjari/HDDFailure](https://github.com/VyshnaviVanjari/HDDFailure)**

**Reach me at Linkedin: [https://www.linkedin.com/in/vyshnavi-vanjari-57345896/](https://www.linkedin.com/in/vyshnavi-vanjari-57345896/)**

