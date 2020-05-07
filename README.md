

```python

```

# Business Case

Analyze customer data to determine any relationships between customer demographic features and the likelihood of a customer purchasing a new bike.  The marketing dept wants to manage their budgets and wisely select the best possible campaigns.  

I will create several Machine Learning models to decipher which demographic data features should be used to target the best possible prospects for new bike purchases. I will compare these models and decide based upon traditional data science model evaluations which one is best option.  The Notebook below has the code that determines what are the best customer demographics.  


```python

```

# OBTAIN

## Imports


```python
import pandas as pd
import glob
```


```python
import os

```


```python
import numpy as np 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.preprocessing import OneHotEncoder
from IPython.display import Image  
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from imblearn.over_sampling import SMOTE

```

    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/externals/six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", FutureWarning)
    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)



```python

```


```python
# showing all cols

pd.set_option('display.max_columns', 0)
```


```python
df= pd.read_csv('combined_data.csv', index_col=0)
```


```python
data_folder = '../bike-buying-prediction-for-adventure-works-cycles/'
files = glob.glob(data_folder+'*csv')
files
```




    ['../bike-buying-prediction-for-adventure-works-cycles/AW_BikeBuyer.csv',
     '../bike-buying-prediction-for-adventure-works-cycles/AdvWorksCusts.csv',
     '../bike-buying-prediction-for-adventure-works-cycles/AW_test.csv',
     '../bike-buying-prediction-for-adventure-works-cycles/AW_AveMonthSpend.csv']




```python
df_buyer = pd.read_csv(files[0])
df_buyer
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>BikeBuyer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>11000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11001</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11002</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>11003</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>11004</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>16514</td>
      <td>13121</td>
      <td>0</td>
    </tr>
    <tr>
      <td>16515</td>
      <td>26100</td>
      <td>0</td>
    </tr>
    <tr>
      <td>16516</td>
      <td>11328</td>
      <td>0</td>
    </tr>
    <tr>
      <td>16517</td>
      <td>23077</td>
      <td>0</td>
    </tr>
    <tr>
      <td>16518</td>
      <td>18982</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16519 rows × 2 columns</p>
</div>




```python
#df.head()
```


```python
#df2 = pd.read_csv(files[-1])
#df2.head()
```


```python
#df = df.merge(df2, on ='CustomerID')
```


```python
# df.to_csv('combined_data.csv')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Title</th>
      <th>FirstName</th>
      <th>MiddleName</th>
      <th>LastName</th>
      <th>Suffix</th>
      <th>AddressLine1</th>
      <th>AddressLine2</th>
      <th>City</th>
      <th>StateProvinceName</th>
      <th>CountryRegionName</th>
      <th>PostalCode</th>
      <th>PhoneNumber</th>
      <th>BirthDate</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>MaritalStatus</th>
      <th>HomeOwnerFlag</th>
      <th>NumberCarsOwned</th>
      <th>NumberChildrenAtHome</th>
      <th>TotalChildren</th>
      <th>YearlyIncome</th>
      <th>target</th>
      <th>AveMonthSpend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>11000</td>
      <td>NaN</td>
      <td>Jon</td>
      <td>V</td>
      <td>Yang</td>
      <td>NaN</td>
      <td>3761 N. 14th St</td>
      <td>NaN</td>
      <td>Rockhampton</td>
      <td>Queensland</td>
      <td>Australia</td>
      <td>4700</td>
      <td>1 (11) 500 555-0162</td>
      <td>1966-04-08</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>137947</td>
      <td>0</td>
      <td>89</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11001</td>
      <td>NaN</td>
      <td>Eugene</td>
      <td>L</td>
      <td>Huang</td>
      <td>NaN</td>
      <td>2243 W St.</td>
      <td>NaN</td>
      <td>Seaford</td>
      <td>Victoria</td>
      <td>Australia</td>
      <td>3198</td>
      <td>1 (11) 500 555-0110</td>
      <td>1965-05-14</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>101141</td>
      <td>1</td>
      <td>117</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11002</td>
      <td>NaN</td>
      <td>Ruben</td>
      <td>NaN</td>
      <td>Torres</td>
      <td>NaN</td>
      <td>5844 Linden Land</td>
      <td>NaN</td>
      <td>Hobart</td>
      <td>Tasmania</td>
      <td>Australia</td>
      <td>7001</td>
      <td>1 (11) 500 555-0184</td>
      <td>1965-08-12</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>91945</td>
      <td>0</td>
      <td>123</td>
    </tr>
    <tr>
      <td>3</td>
      <td>11003</td>
      <td>NaN</td>
      <td>Christy</td>
      <td>NaN</td>
      <td>Zhu</td>
      <td>NaN</td>
      <td>1825 Village Pl.</td>
      <td>NaN</td>
      <td>North Ryde</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>2113</td>
      <td>1 (11) 500 555-0162</td>
      <td>1968-02-15</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>86688</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <td>4</td>
      <td>11004</td>
      <td>NaN</td>
      <td>Elizabeth</td>
      <td>NaN</td>
      <td>Johnson</td>
      <td>NaN</td>
      <td>7553 Harness Circle</td>
      <td>NaN</td>
      <td>Wollongong</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>2500</td>
      <td>1 (11) 500 555-0131</td>
      <td>1968-08-08</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>92771</td>
      <td>1</td>
      <td>95</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>16744</td>
      <td>29478</td>
      <td>NaN</td>
      <td>Darren</td>
      <td>D</td>
      <td>Carlson</td>
      <td>NaN</td>
      <td>5240 Premier Pl.</td>
      <td>NaN</td>
      <td>Stoke-on-Trent</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>AS23</td>
      <td>1 (11) 500 555-0132</td>
      <td>1959-05-25</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>45986</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <td>16745</td>
      <td>29479</td>
      <td>NaN</td>
      <td>Tommy</td>
      <td>L</td>
      <td>Tang</td>
      <td>NaN</td>
      <td>111, rue Maillard</td>
      <td>NaN</td>
      <td>Versailles</td>
      <td>Yveline</td>
      <td>France</td>
      <td>78000</td>
      <td>1 (11) 500 555-0136</td>
      <td>1958-07-04</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>80049</td>
      <td>0</td>
      <td>77</td>
    </tr>
    <tr>
      <td>16746</td>
      <td>29480</td>
      <td>NaN</td>
      <td>Nina</td>
      <td>W</td>
      <td>Raji</td>
      <td>NaN</td>
      <td>9 Katherine Drive</td>
      <td>NaN</td>
      <td>London</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>SW19 3RU</td>
      <td>1 (11) 500 555-0146</td>
      <td>1960-11-10</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>60417</td>
      <td>1</td>
      <td>48</td>
    </tr>
    <tr>
      <td>16747</td>
      <td>29481</td>
      <td>NaN</td>
      <td>Ivan</td>
      <td>NaN</td>
      <td>Suri</td>
      <td>NaN</td>
      <td>Knaackstr 4</td>
      <td>NaN</td>
      <td>Hof</td>
      <td>Bayern</td>
      <td>Germany</td>
      <td>95010</td>
      <td>1 (11) 500 555-0144</td>
      <td>1960-01-05</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>66653</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <td>16748</td>
      <td>29482</td>
      <td>NaN</td>
      <td>Clayton</td>
      <td>NaN</td>
      <td>Zhang</td>
      <td>NaN</td>
      <td>1080, quai de Grenelle</td>
      <td>NaN</td>
      <td>Saint Ouen</td>
      <td>Charente-Maritime</td>
      <td>France</td>
      <td>17490</td>
      <td>1 (11) 500 555-0137</td>
      <td>1959-03-05</td>
      <td>Bachelors</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>59736</td>
      <td>0</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
<p>16749 rows × 25 columns</p>
</div>




```python

```

# SCRUB/EXPLORE

### Initial DataFrame


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Title</th>
      <th>FirstName</th>
      <th>MiddleName</th>
      <th>LastName</th>
      <th>Suffix</th>
      <th>AddressLine1</th>
      <th>AddressLine2</th>
      <th>City</th>
      <th>StateProvinceName</th>
      <th>CountryRegionName</th>
      <th>PostalCode</th>
      <th>PhoneNumber</th>
      <th>BirthDate</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>MaritalStatus</th>
      <th>HomeOwnerFlag</th>
      <th>NumberCarsOwned</th>
      <th>NumberChildrenAtHome</th>
      <th>TotalChildren</th>
      <th>YearlyIncome</th>
      <th>target</th>
      <th>AveMonthSpend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>11000</td>
      <td>NaN</td>
      <td>Jon</td>
      <td>V</td>
      <td>Yang</td>
      <td>NaN</td>
      <td>3761 N. 14th St</td>
      <td>NaN</td>
      <td>Rockhampton</td>
      <td>Queensland</td>
      <td>Australia</td>
      <td>4700</td>
      <td>1 (11) 500 555-0162</td>
      <td>1966-04-08</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>137947</td>
      <td>0</td>
      <td>89</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11001</td>
      <td>NaN</td>
      <td>Eugene</td>
      <td>L</td>
      <td>Huang</td>
      <td>NaN</td>
      <td>2243 W St.</td>
      <td>NaN</td>
      <td>Seaford</td>
      <td>Victoria</td>
      <td>Australia</td>
      <td>3198</td>
      <td>1 (11) 500 555-0110</td>
      <td>1965-05-14</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>101141</td>
      <td>1</td>
      <td>117</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11002</td>
      <td>NaN</td>
      <td>Ruben</td>
      <td>NaN</td>
      <td>Torres</td>
      <td>NaN</td>
      <td>5844 Linden Land</td>
      <td>NaN</td>
      <td>Hobart</td>
      <td>Tasmania</td>
      <td>Australia</td>
      <td>7001</td>
      <td>1 (11) 500 555-0184</td>
      <td>1965-08-12</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>91945</td>
      <td>0</td>
      <td>123</td>
    </tr>
    <tr>
      <td>3</td>
      <td>11003</td>
      <td>NaN</td>
      <td>Christy</td>
      <td>NaN</td>
      <td>Zhu</td>
      <td>NaN</td>
      <td>1825 Village Pl.</td>
      <td>NaN</td>
      <td>North Ryde</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>2113</td>
      <td>1 (11) 500 555-0162</td>
      <td>1968-02-15</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>86688</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <td>4</td>
      <td>11004</td>
      <td>NaN</td>
      <td>Elizabeth</td>
      <td>NaN</td>
      <td>Johnson</td>
      <td>NaN</td>
      <td>7553 Harness Circle</td>
      <td>NaN</td>
      <td>Wollongong</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>2500</td>
      <td>1 (11) 500 555-0131</td>
      <td>1968-08-08</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>92771</td>
      <td>1</td>
      <td>95</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>16744</td>
      <td>29478</td>
      <td>NaN</td>
      <td>Darren</td>
      <td>D</td>
      <td>Carlson</td>
      <td>NaN</td>
      <td>5240 Premier Pl.</td>
      <td>NaN</td>
      <td>Stoke-on-Trent</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>AS23</td>
      <td>1 (11) 500 555-0132</td>
      <td>1959-05-25</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>45986</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <td>16745</td>
      <td>29479</td>
      <td>NaN</td>
      <td>Tommy</td>
      <td>L</td>
      <td>Tang</td>
      <td>NaN</td>
      <td>111, rue Maillard</td>
      <td>NaN</td>
      <td>Versailles</td>
      <td>Yveline</td>
      <td>France</td>
      <td>78000</td>
      <td>1 (11) 500 555-0136</td>
      <td>1958-07-04</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>80049</td>
      <td>0</td>
      <td>77</td>
    </tr>
    <tr>
      <td>16746</td>
      <td>29480</td>
      <td>NaN</td>
      <td>Nina</td>
      <td>W</td>
      <td>Raji</td>
      <td>NaN</td>
      <td>9 Katherine Drive</td>
      <td>NaN</td>
      <td>London</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>SW19 3RU</td>
      <td>1 (11) 500 555-0146</td>
      <td>1960-11-10</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>60417</td>
      <td>1</td>
      <td>48</td>
    </tr>
    <tr>
      <td>16747</td>
      <td>29481</td>
      <td>NaN</td>
      <td>Ivan</td>
      <td>NaN</td>
      <td>Suri</td>
      <td>NaN</td>
      <td>Knaackstr 4</td>
      <td>NaN</td>
      <td>Hof</td>
      <td>Bayern</td>
      <td>Germany</td>
      <td>95010</td>
      <td>1 (11) 500 555-0144</td>
      <td>1960-01-05</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>66653</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <td>16748</td>
      <td>29482</td>
      <td>NaN</td>
      <td>Clayton</td>
      <td>NaN</td>
      <td>Zhang</td>
      <td>NaN</td>
      <td>1080, quai de Grenelle</td>
      <td>NaN</td>
      <td>Saint Ouen</td>
      <td>Charente-Maritime</td>
      <td>France</td>
      <td>17490</td>
      <td>1 (11) 500 555-0137</td>
      <td>1959-03-05</td>
      <td>Bachelors</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>59736</td>
      <td>0</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
<p>16749 rows × 25 columns</p>
</div>



## Target 

### Target Value Counts (Ones are Buyers)
0 are non-buyers (67%)
1 are buyers (33%)



```python
df['target'].value_counts(normalize=True)
```




    0    0.66798
    1    0.33202
    Name: target, dtype: float64



## Null Values


```python
df.isnull().sum().divide(len(df))
```




    CustomerID              0.000000
    Title                   0.994746
    FirstName               0.000000
    MiddleName              0.421100
    LastName                0.000000
    Suffix                  0.999881
    AddressLine1            0.000000
    AddressLine2            0.983223
    City                    0.000000
    StateProvinceName       0.000000
    CountryRegionName       0.000000
    PostalCode              0.000000
    PhoneNumber             0.000000
    BirthDate               0.000000
    Education               0.000000
    Occupation              0.000000
    Gender                  0.000000
    MaritalStatus           0.000000
    HomeOwnerFlag           0.000000
    NumberCarsOwned         0.000000
    NumberChildrenAtHome    0.000000
    TotalChildren           0.000000
    YearlyIncome            0.000000
    target                  0.000000
    AveMonthSpend           0.000000
    dtype: float64



## Drop Columns


```python
df.drop(['Title', 'FirstName', 'MiddleName', 
         'LastName', 'Suffix', 'AddressLine1', 
         'AddressLine2', 'PhoneNumber', 'PostalCode'], axis=1, inplace=True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>City</th>
      <th>StateProvinceName</th>
      <th>CountryRegionName</th>
      <th>BirthDate</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>MaritalStatus</th>
      <th>HomeOwnerFlag</th>
      <th>NumberCarsOwned</th>
      <th>NumberChildrenAtHome</th>
      <th>TotalChildren</th>
      <th>YearlyIncome</th>
      <th>target</th>
      <th>AveMonthSpend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>11000</td>
      <td>Rockhampton</td>
      <td>Queensland</td>
      <td>Australia</td>
      <td>1966-04-08</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>137947</td>
      <td>0</td>
      <td>89</td>
    </tr>
    <tr>
      <td>1</td>
      <td>11001</td>
      <td>Seaford</td>
      <td>Victoria</td>
      <td>Australia</td>
      <td>1965-05-14</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>101141</td>
      <td>1</td>
      <td>117</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11002</td>
      <td>Hobart</td>
      <td>Tasmania</td>
      <td>Australia</td>
      <td>1965-08-12</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>91945</td>
      <td>0</td>
      <td>123</td>
    </tr>
    <tr>
      <td>3</td>
      <td>11003</td>
      <td>North Ryde</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>1968-02-15</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>86688</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <td>4</td>
      <td>11004</td>
      <td>Wollongong</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>1968-08-08</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>92771</td>
      <td>1</td>
      <td>95</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>16744</td>
      <td>29478</td>
      <td>Stoke-on-Trent</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>1959-05-25</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>45986</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <td>16745</td>
      <td>29479</td>
      <td>Versailles</td>
      <td>Yveline</td>
      <td>France</td>
      <td>1958-07-04</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>80049</td>
      <td>0</td>
      <td>77</td>
    </tr>
    <tr>
      <td>16746</td>
      <td>29480</td>
      <td>London</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>1960-11-10</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>60417</td>
      <td>1</td>
      <td>48</td>
    </tr>
    <tr>
      <td>16747</td>
      <td>29481</td>
      <td>Hof</td>
      <td>Bayern</td>
      <td>Germany</td>
      <td>1960-01-05</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>66653</td>
      <td>0</td>
      <td>65</td>
    </tr>
    <tr>
      <td>16748</td>
      <td>29482</td>
      <td>Saint Ouen</td>
      <td>Charente-Maritime</td>
      <td>France</td>
      <td>1959-03-05</td>
      <td>Bachelors</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>59736</td>
      <td>0</td>
      <td>72</td>
    </tr>
  </tbody>
</table>
<p>16749 rows × 16 columns</p>
</div>




```python
#should have removed all null values
df.isnull().sum()
```




    CustomerID              0
    City                    0
    StateProvinceName       0
    CountryRegionName       0
    BirthDate               0
    Education               0
    Occupation              0
    Gender                  0
    MaritalStatus           0
    HomeOwnerFlag           0
    NumberCarsOwned         0
    NumberChildrenAtHome    0
    TotalChildren           0
    YearlyIncome            0
    target                  0
    AveMonthSpend           0
    dtype: int64



## Find Categorical & Numerical Values


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16749 entries, 0 to 16748
    Data columns (total 16 columns):
    CustomerID              16749 non-null int64
    City                    16749 non-null object
    StateProvinceName       16749 non-null object
    CountryRegionName       16749 non-null object
    BirthDate               16749 non-null object
    Education               16749 non-null object
    Occupation              16749 non-null object
    Gender                  16749 non-null object
    MaritalStatus           16749 non-null object
    HomeOwnerFlag           16749 non-null int64
    NumberCarsOwned         16749 non-null int64
    NumberChildrenAtHome    16749 non-null int64
    TotalChildren           16749 non-null int64
    YearlyIncome            16749 non-null int64
    target                  16749 non-null int64
    AveMonthSpend           16749 non-null int64
    dtypes: int64(8), object(8)
    memory usage: 2.2+ MB



```python
#set index to customer id
df.set_index('CustomerID', inplace=True)
```


```python
df['BirthDate'] = df['BirthDate'].str.split("-").apply(lambda x: x[0]).astype(int)
```


```python
df['BirthDate']
```




    CustomerID
    11000    1966
    11001    1965
    11002    1965
    11003    1968
    11004    1968
             ... 
    29478    1959
    29479    1958
    29480    1960
    29481    1960
    29482    1959
    Name: BirthDate, Length: 16749, dtype: int64




```python
df['Age'] = 2020 - df['BirthDate']
df.drop('BirthDate' , inplace=True, axis=1)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>StateProvinceName</th>
      <th>CountryRegionName</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>MaritalStatus</th>
      <th>HomeOwnerFlag</th>
      <th>NumberCarsOwned</th>
      <th>NumberChildrenAtHome</th>
      <th>TotalChildren</th>
      <th>YearlyIncome</th>
      <th>target</th>
      <th>AveMonthSpend</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>11000</td>
      <td>Rockhampton</td>
      <td>Queensland</td>
      <td>Australia</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>137947</td>
      <td>0</td>
      <td>89</td>
      <td>54</td>
    </tr>
    <tr>
      <td>11001</td>
      <td>Seaford</td>
      <td>Victoria</td>
      <td>Australia</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>101141</td>
      <td>1</td>
      <td>117</td>
      <td>55</td>
    </tr>
    <tr>
      <td>11002</td>
      <td>Hobart</td>
      <td>Tasmania</td>
      <td>Australia</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>91945</td>
      <td>0</td>
      <td>123</td>
      <td>55</td>
    </tr>
    <tr>
      <td>11003</td>
      <td>North Ryde</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>86688</td>
      <td>0</td>
      <td>50</td>
      <td>52</td>
    </tr>
    <tr>
      <td>11004</td>
      <td>Wollongong</td>
      <td>New South Wales</td>
      <td>Australia</td>
      <td>Bachelors</td>
      <td>Professional</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>92771</td>
      <td>1</td>
      <td>95</td>
      <td>52</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>29478</td>
      <td>Stoke-on-Trent</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>45986</td>
      <td>0</td>
      <td>65</td>
      <td>61</td>
    </tr>
    <tr>
      <td>29479</td>
      <td>Versailles</td>
      <td>Yveline</td>
      <td>France</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>80049</td>
      <td>0</td>
      <td>77</td>
      <td>62</td>
    </tr>
    <tr>
      <td>29480</td>
      <td>London</td>
      <td>England</td>
      <td>United Kingdom</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>F</td>
      <td>S</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>60417</td>
      <td>1</td>
      <td>48</td>
      <td>60</td>
    </tr>
    <tr>
      <td>29481</td>
      <td>Hof</td>
      <td>Bayern</td>
      <td>Germany</td>
      <td>Graduate Degree</td>
      <td>Clerical</td>
      <td>M</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>66653</td>
      <td>0</td>
      <td>65</td>
      <td>60</td>
    </tr>
    <tr>
      <td>29482</td>
      <td>Saint Ouen</td>
      <td>Charente-Maritime</td>
      <td>France</td>
      <td>Bachelors</td>
      <td>Clerical</td>
      <td>M</td>
      <td>M</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>59736</td>
      <td>0</td>
      <td>72</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
<p>16749 rows × 15 columns</p>
</div>




```python

```

### Seperate categoricals vs numbers


```python
# Categorical 
catag_cols = df.select_dtypes('object').columns
```


```python
catag_cols
```




    Index(['City', 'StateProvinceName', 'CountryRegionName', 'Education',
           'Occupation', 'Gender', 'MaritalStatus'],
          dtype='object')




```python

```


```python
#Number cols to scale
number_cols = df.select_dtypes('number').columns

```


```python
number_cols
```




    Index(['HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome',
           'TotalChildren', 'YearlyIncome', 'target', 'AveMonthSpend', 'Age'],
          dtype='object')




```python
# Finding unique categorical 
for col in catag_cols:
    print(col)
    print(df[col].nunique())
```

    City
    270
    StateProvinceName
    52
    CountryRegionName
    6
    Education
    5
    Occupation
    5
    Gender
    2
    MaritalStatus
    2


### One Hot Encoding


```python
df = pd.get_dummies(df, columns=catag_cols)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HomeOwnerFlag</th>
      <th>NumberCarsOwned</th>
      <th>NumberChildrenAtHome</th>
      <th>TotalChildren</th>
      <th>YearlyIncome</th>
      <th>target</th>
      <th>AveMonthSpend</th>
      <th>Age</th>
      <th>City_Ballard</th>
      <th>City_Baltimore</th>
      <th>City_Barstow</th>
      <th>City_Basingstoke Hants</th>
      <th>City_Baytown</th>
      <th>City_Beaverton</th>
      <th>City_Bell Gardens</th>
      <th>City_Bellevue</th>
      <th>City_Bellflower</th>
      <th>City_Bellingham</th>
      <th>City_Bendigo</th>
      <th>City_Berkeley</th>
      <th>City_Berks</th>
      <th>City_Berkshire</th>
      <th>City_Berlin</th>
      <th>City_Beverly Hills</th>
      <th>City_Billericay</th>
      <th>City_Biloxi</th>
      <th>City_Birmingham</th>
      <th>City_Bluffton</th>
      <th>City_Bobigny</th>
      <th>City_Bonn</th>
      <th>City_Bothell</th>
      <th>City_Bottrop</th>
      <th>City_Boulogne-Billancourt</th>
      <th>City_Boulogne-sur-Mer</th>
      <th>City_Bountiful</th>
      <th>City_Bracknell</th>
      <th>City_Bradenton</th>
      <th>City_Braintree</th>
      <th>City_Branch</th>
      <th>City_Branson</th>
      <th>City_Braunschweig</th>
      <th>City_Bremerton</th>
      <th>City_Brisbane</th>
      <th>City_Burbank</th>
      <th>City_Burien</th>
      <th>City_Burlingame</th>
      <th>City_Burnaby</th>
      <th>City_Bury</th>
      <th>City_Byron</th>
      <th>City_Calgary</th>
      <th>...</th>
      <th>StateProvinceName_Minnesota</th>
      <th>StateProvinceName_Mississippi</th>
      <th>StateProvinceName_Missouri</th>
      <th>StateProvinceName_Moselle</th>
      <th>StateProvinceName_New South Wales</th>
      <th>StateProvinceName_New York</th>
      <th>StateProvinceName_Nord</th>
      <th>StateProvinceName_Nordrhein-Westfalen</th>
      <th>StateProvinceName_North Carolina</th>
      <th>StateProvinceName_Ohio</th>
      <th>StateProvinceName_Oregon</th>
      <th>StateProvinceName_Pas de Calais</th>
      <th>StateProvinceName_Queensland</th>
      <th>StateProvinceName_Saarland</th>
      <th>StateProvinceName_Seine (Paris)</th>
      <th>StateProvinceName_Seine Saint Denis</th>
      <th>StateProvinceName_Seine et Marne</th>
      <th>StateProvinceName_Somme</th>
      <th>StateProvinceName_South Australia</th>
      <th>StateProvinceName_South Carolina</th>
      <th>StateProvinceName_Tasmania</th>
      <th>StateProvinceName_Texas</th>
      <th>StateProvinceName_Utah</th>
      <th>StateProvinceName_Val d'Oise</th>
      <th>StateProvinceName_Val de Marne</th>
      <th>StateProvinceName_Victoria</th>
      <th>StateProvinceName_Virginia</th>
      <th>StateProvinceName_Washington</th>
      <th>StateProvinceName_Wyoming</th>
      <th>StateProvinceName_Yveline</th>
      <th>CountryRegionName_Australia</th>
      <th>CountryRegionName_Canada</th>
      <th>CountryRegionName_France</th>
      <th>CountryRegionName_Germany</th>
      <th>CountryRegionName_United Kingdom</th>
      <th>CountryRegionName_United States</th>
      <th>Education_Bachelors</th>
      <th>Education_Graduate Degree</th>
      <th>Education_High School</th>
      <th>Education_Partial College</th>
      <th>Education_Partial High School</th>
      <th>Occupation_Clerical</th>
      <th>Occupation_Management</th>
      <th>Occupation_Manual</th>
      <th>Occupation_Professional</th>
      <th>Occupation_Skilled Manual</th>
      <th>Gender_F</th>
      <th>Gender_M</th>
      <th>MaritalStatus_M</th>
      <th>MaritalStatus_S</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>11000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>137947</td>
      <td>0</td>
      <td>89</td>
      <td>54</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>11001</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>101141</td>
      <td>1</td>
      <td>117</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11002</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>91945</td>
      <td>0</td>
      <td>123</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>11003</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>86688</td>
      <td>0</td>
      <td>50</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11004</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>92771</td>
      <td>1</td>
      <td>95</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 350 columns</p>
</div>




```python

```

## Scale Data


```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```


```python
scaler = StandardScaler()
```


```python
df_scaled = df.copy()
```


```python
df_scaled[number_cols] = scaler.fit_transform(df_scaled[number_cols])
```


```python
df_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HomeOwnerFlag</th>
      <th>NumberCarsOwned</th>
      <th>NumberChildrenAtHome</th>
      <th>TotalChildren</th>
      <th>YearlyIncome</th>
      <th>target</th>
      <th>AveMonthSpend</th>
      <th>Age</th>
      <th>City_Ballard</th>
      <th>City_Baltimore</th>
      <th>City_Barstow</th>
      <th>City_Basingstoke Hants</th>
      <th>City_Baytown</th>
      <th>City_Beaverton</th>
      <th>City_Bell Gardens</th>
      <th>City_Bellevue</th>
      <th>City_Bellflower</th>
      <th>City_Bellingham</th>
      <th>City_Bendigo</th>
      <th>City_Berkeley</th>
      <th>City_Berks</th>
      <th>City_Berkshire</th>
      <th>City_Berlin</th>
      <th>City_Beverly Hills</th>
      <th>City_Billericay</th>
      <th>City_Biloxi</th>
      <th>City_Birmingham</th>
      <th>City_Bluffton</th>
      <th>City_Bobigny</th>
      <th>City_Bonn</th>
      <th>City_Bothell</th>
      <th>City_Bottrop</th>
      <th>City_Boulogne-Billancourt</th>
      <th>City_Boulogne-sur-Mer</th>
      <th>City_Bountiful</th>
      <th>City_Bracknell</th>
      <th>City_Bradenton</th>
      <th>City_Braintree</th>
      <th>City_Branch</th>
      <th>City_Branson</th>
      <th>City_Braunschweig</th>
      <th>City_Bremerton</th>
      <th>City_Brisbane</th>
      <th>City_Burbank</th>
      <th>City_Burien</th>
      <th>City_Burlingame</th>
      <th>City_Burnaby</th>
      <th>City_Bury</th>
      <th>City_Byron</th>
      <th>City_Calgary</th>
      <th>...</th>
      <th>StateProvinceName_Minnesota</th>
      <th>StateProvinceName_Mississippi</th>
      <th>StateProvinceName_Missouri</th>
      <th>StateProvinceName_Moselle</th>
      <th>StateProvinceName_New South Wales</th>
      <th>StateProvinceName_New York</th>
      <th>StateProvinceName_Nord</th>
      <th>StateProvinceName_Nordrhein-Westfalen</th>
      <th>StateProvinceName_North Carolina</th>
      <th>StateProvinceName_Ohio</th>
      <th>StateProvinceName_Oregon</th>
      <th>StateProvinceName_Pas de Calais</th>
      <th>StateProvinceName_Queensland</th>
      <th>StateProvinceName_Saarland</th>
      <th>StateProvinceName_Seine (Paris)</th>
      <th>StateProvinceName_Seine Saint Denis</th>
      <th>StateProvinceName_Seine et Marne</th>
      <th>StateProvinceName_Somme</th>
      <th>StateProvinceName_South Australia</th>
      <th>StateProvinceName_South Carolina</th>
      <th>StateProvinceName_Tasmania</th>
      <th>StateProvinceName_Texas</th>
      <th>StateProvinceName_Utah</th>
      <th>StateProvinceName_Val d'Oise</th>
      <th>StateProvinceName_Val de Marne</th>
      <th>StateProvinceName_Victoria</th>
      <th>StateProvinceName_Virginia</th>
      <th>StateProvinceName_Washington</th>
      <th>StateProvinceName_Wyoming</th>
      <th>StateProvinceName_Yveline</th>
      <th>CountryRegionName_Australia</th>
      <th>CountryRegionName_Canada</th>
      <th>CountryRegionName_France</th>
      <th>CountryRegionName_Germany</th>
      <th>CountryRegionName_United Kingdom</th>
      <th>CountryRegionName_United States</th>
      <th>Education_Bachelors</th>
      <th>Education_Graduate Degree</th>
      <th>Education_High School</th>
      <th>Education_Partial College</th>
      <th>Education_Partial High School</th>
      <th>Occupation_Clerical</th>
      <th>Occupation_Management</th>
      <th>Occupation_Manual</th>
      <th>Occupation_Professional</th>
      <th>Occupation_Skilled Manual</th>
      <th>Gender_F</th>
      <th>Gender_M</th>
      <th>MaritalStatus_M</th>
      <th>MaritalStatus_S</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>11000</td>
      <td>0.696305</td>
      <td>-1.320438</td>
      <td>-0.655315</td>
      <td>-0.005710</td>
      <td>1.508094</td>
      <td>-0.705018</td>
      <td>0.606392</td>
      <td>-0.398053</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>11001</td>
      <td>-1.436153</td>
      <td>-0.442156</td>
      <td>1.322913</td>
      <td>0.588291</td>
      <td>0.580465</td>
      <td>1.418403</td>
      <td>1.631466</td>
      <td>-0.309197</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11002</td>
      <td>0.696305</td>
      <td>-0.442156</td>
      <td>1.322913</td>
      <td>0.588291</td>
      <td>0.348696</td>
      <td>-0.705018</td>
      <td>1.851125</td>
      <td>-0.309197</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>11003</td>
      <td>-1.436153</td>
      <td>-0.442156</td>
      <td>-0.655315</td>
      <td>-1.193712</td>
      <td>0.216203</td>
      <td>-0.705018</td>
      <td>-0.821389</td>
      <td>-0.575763</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11004</td>
      <td>0.696305</td>
      <td>2.192690</td>
      <td>2.641731</td>
      <td>1.776294</td>
      <td>0.369514</td>
      <td>1.418403</td>
      <td>0.826051</td>
      <td>-0.575763</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>29478</td>
      <td>0.696305</td>
      <td>-1.320438</td>
      <td>-0.655315</td>
      <td>0.588291</td>
      <td>-0.809617</td>
      <td>-0.705018</td>
      <td>-0.272242</td>
      <td>0.223934</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>29479</td>
      <td>0.696305</td>
      <td>-1.320438</td>
      <td>-0.655315</td>
      <td>-0.599711</td>
      <td>0.048879</td>
      <td>-0.705018</td>
      <td>0.167075</td>
      <td>0.312789</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>29480</td>
      <td>0.696305</td>
      <td>-1.320438</td>
      <td>-0.655315</td>
      <td>0.588291</td>
      <td>-0.445910</td>
      <td>1.418403</td>
      <td>-0.894608</td>
      <td>0.135079</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>29481</td>
      <td>-1.436153</td>
      <td>-1.320438</td>
      <td>-0.655315</td>
      <td>0.588291</td>
      <td>-0.288743</td>
      <td>-0.705018</td>
      <td>-0.272242</td>
      <td>0.135079</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>29482</td>
      <td>0.696305</td>
      <td>-1.320438</td>
      <td>-0.655315</td>
      <td>0.588291</td>
      <td>-0.463073</td>
      <td>-0.705018</td>
      <td>-0.015974</td>
      <td>0.223934</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16749 rows × 350 columns</p>
</div>




```python

```

# MODEL

 ## Train Test Split


```python
# Create features and label/target
X = df.drop('target', axis=1)  
y = df['target'] 
```


```python
# An 80/20 split with random_state of 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

```

### SMOTE


```python
smote = SMOTE()
X_train, y_train = smote.fit_sample(X_train, y_train)
```

    //anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
      warnings.warn(msg, category=FutureWarning)



```python
X_train =pd.DataFrame(X_train, columns=X.columns)
```

## First Model: Decision Tree


```python

```


```python
# Train Decesion Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=10)  
dt_classifier.fit(X_train, y_train) 
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=10, splitter='best')




```python

```


```python
# Make predictions for test data
y_pred = dt_classifier.predict(X_test)
```

### Evaluate Model
Accuracy, AUC & Confusion Matrix Results


```python
def evaluate_model(X_test, y_test, classifier):

    from sklearn.metrics import plot_confusion_matrix

    # Make predictions for test data
    y_pred = classifier.predict(X_test)

    # Calculate accuracy 
    acc = accuracy_score(y_test,y_pred) * 100
    print('Accuracy is :{0}'.format(round(acc, 2)))

    # Check the AUC for predictions
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print('\nAUC is :{0}'.format(round(roc_auc, 2)))

    # Create and print a confusion matrix 
    print('\nConfusion Matrix')
    print('----------------')
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

    # Create Confusion Matrix
    plot_confusion_matrix(classifier, X_test, y_test, normalize='true',
                          cmap='Blues', display_labels= ['NonBuyers', 'Buyers'])
    plt.show()
```


```python
evaluate_model(X_test, y_test, dt_classifier)
```

    Accuracy is :74.9
    
    AUC is :0.72
    
    Confusion Matrix
    ----------------



![png](output_71_1.png)



```python

```

### Feature Importance Visualization: Decision Tree


```python
def plot_feature_importances(model):
    n_features = X_train.shape[1]
    sort = pd.Series(model.feature_importances_,index=X_train.columns).sort_values(ascending=False).head(10)
    sort.plot(kind ='barh')
    #plt.figure(figsize=(12,60))
    #plt.barh(range(n_features), sort, align='center') 
    #plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.title('Feature Importance')
    plt.ylabel('Feature')
    
    plt.show()
    
    return sort

#  Bar graph with corr matrix 
    
def get_corr(df, model):
    sort = plot_feature_importances(model)
    fig, ax = plt.subplots(figsize=(5,5))
    df_corr = df.corr()['target']
    df_corr.loc[sort.index].plot(kind='barh', ax=ax)
    ax.set_xlabel('Correlation with Buyer', fontdict=dict(size=12))
           
    ax.set_title('Correlation of Most Important Features', fontdict=dict(size=14))
    ax.axvline(0.0)
    plt.show()
    
    #return fig

```


```python

```

###  GridsearchCV


```python
# Perform a 3-fold cross-validation on the training data 
# using the dt_classifier (from last section)

dt_cv_score = cross_val_score(dt_classifier, X_train, y_train, cv=3)
mean_dt_cv_score = np.mean(dt_cv_score)

print(f"Mean Cross Validation Score: {mean_dt_cv_score :.2%}")
```

    Mean Cross Validation Score: 77.86%



```python

```


```python
# Need to create a dictionary for Combinatoric Grid Searching

dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]
}

# Instantiate GridSearchCV
dt_grid_search = GridSearchCV(dt_classifier, dt_param_grid, cv=3, return_train_score=True)

# Fit to the data
dt_grid_search.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=10,
                                                  splitter='best'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'criterion': ['gini', 'entropy'],
                             'max_depth': [None, 2, 3, 4, 5, 6],
                             'min_samples_leaf': [1, 2, 3, 4, 5, 6],
                             'min_samples_split': [2, 5, 10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
                 scoring=None, verbose=0)




```python
dt_grid_search.best_params_
```




    {'criterion': 'gini',
     'max_depth': None,
     'min_samples_leaf': 5,
     'min_samples_split': 2}




```python
dt_best_model = dt_grid_search.best_estimator_
```


```python
# Mean training score
dt_gs_training_score = dt_best_model.score(X_train, y_train)

# Mean test score
dt_gs_testing_score = dt_best_model.score(X_test, y_test)

print(f"Mean Training Score: {dt_gs_training_score :.2%}")
print(f"Mean Test Score: {dt_gs_testing_score :.2%}")
print("Best Parameter Combination Found During Grid Search:")
dt_grid_search.best_params_
```

    Mean Training Score: 90.50%
    Mean Test Score: 74.18%
    Best Parameter Combination Found During Grid Search:





    {'criterion': 'gini',
     'max_depth': None,
     'min_samples_leaf': 5,
     'min_samples_split': 2}



## Decision Tree Graphical Representation


```python
## visualize the decision tree
def visualize_tree(tree,feature_names=None,class_names=['0','1'],export_graphviz_kws={}):
    """Visualizes a sklearn tree using sklearn.tree.export_graphviz"""
    from sklearn.tree import export_graphviz
    from IPython.display import SVG
    from graphviz import Source
    from IPython.display import display
    if feature_names is None:
        feature_names=X_train.columns

    tree_viz_kws =  dict(out_file=None, rotate=False, filled = True)
    tree_viz_kws.update(export_graphviz_kws)

    # tree.export_graphviz(dt) #if you wish to save the output to a dot file instead
    graph = Source(export_graphviz(tree,feature_names=feature_names, class_names=class_names,**tree_viz_kws))
    display(SVG(graph.pipe(format='svg')))
```


```python
visualize_tree(dt_best_model)
```


![svg](output_85_0.svg)


### After Gridsearch: Feature Importance Visuals 


```python
evaluate_model(X_test, y_test, dt_best_model)
```

    Accuracy is :74.18
    
    AUC is :0.71
    
    Confusion Matrix
    ----------------



![png](output_87_1.png)



```python

```


```python
get_corr(df, dt_best_model)
```


![png](output_89_0.png)



![png](output_89_1.png)



```python

```

## Second Model: Bagging Classifier

### Imports for Model


```python
np.random.seed(0)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
```

### BaggingClassifier


```python
# Instantiate a BaggingClassifier
bagged_tree =  BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5), 
                                 n_estimators=20)
```


```python

```


```python
# Fit to the training data
bagged_tree.fit(X_train, y_train)
```




    BaggingClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.0,
                                                            class_weight=None,
                                                            criterion='gini',
                                                            max_depth=5,
                                                            max_features=None,
                                                            max_leaf_nodes=None,
                                                            min_impurity_decrease=0.0,
                                                            min_impurity_split=None,
                                                            min_samples_leaf=1,
                                                            min_samples_split=2,
                                                            min_weight_fraction_leaf=0.0,
                                                            presort='deprecated',
                                                            random_state=None,
                                                            splitter='best'),
                      bootstrap=True, bootstrap_features=False, max_features=1.0,
                      max_samples=1.0, n_estimators=20, n_jobs=None,
                      oob_score=False, random_state=None, verbose=0,
                      warm_start=False)



### BaggingClassifier Training Accuracy Score


```python
# Training accuracy score
bagged_tree.score(X_train, y_train)
```




    0.7911307460015659



### BaggingClassifier Test Accuracy Score
(the accuracy score that really matters) 




```python
# Test accuracy score
bagged_tree.score(X_test, y_test)
```




    0.7823880597014925




```python

```

## Third Model:  Random Forest


```python
#Forest Grid Search

dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]
}

# Instantiate GridSearchCV
forest_grid_search = GridSearchCV(RandomForestClassifier(n_estimators=100), dt_param_grid, cv=3, return_train_score=True)

# Fit to the data
forest_grid_search.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score=nan,
                 estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                                  class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  max_samples=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators=100, n_jobs=None,
                                                  oob_score=False,
                                                  random_state=None, verbose=0,
                                                  warm_start=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'criterion': ['gini', 'entropy'],
                             'max_depth': [2, 3, 4, 5, 6],
                             'min_samples_leaf': [1, 2, 3, 4, 5, 6],
                             'min_samples_split': [2, 5, 10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
                 scoring=None, verbose=0)




```python

```


```python
# Instantiate and fit a RandomForestClassifier
forest = forest_grid_search.best_estimator_
forest.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='entropy', max_depth=6, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=4, min_samples_split=10,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



### Training Accuracy Score


```python
# Training accuracy score
forest.score(X_train, y_train)
```




    0.8112068001342132



### Test Accuracy Score


```python
# Test accuracy score
forest.score(X_test, y_test)
```




    0.7829850746268656



### Evaluate Model
Accuracy, AUC & Confusion Matrix Results


```python
evaluate_model(X_test, y_test, forest)
```

    Accuracy is :78.3
    
    AUC is :0.76
    
    Confusion Matrix
    ----------------



![png](output_112_1.png)


### Feature Importance Visualization


```python
plot_feature_importances(forest)
```


![png](output_114_0.png)





    NumberChildrenAtHome               0.156870
    AveMonthSpend                      0.146077
    MaritalStatus_M                    0.106652
    Age                                0.081407
    YearlyIncome                       0.081257
    TotalChildren                      0.048523
    Gender_F                           0.046884
    NumberCarsOwned                    0.040877
    CountryRegionName_United States    0.028144
    MaritalStatus_S                    0.020671
    dtype: float64




```python
get_corr(df, forest)
```


![png](output_115_0.png)



![png](output_115_1.png)



```python

```



## Fourth Model: XGBoost

### Imports for this model


```python
from xgboost import XGBClassifier
```

### Fit
### Predict
### Training and Test Accuracy Scores


```python
# Instantiate XGBClassifier
xgb = XGBClassifier()

# Fit XGBClassifier
xgb.fit(X_train, y_train)

# Predict on training and test sets
training_preds = xgb.predict(X_train)
test_preds = xgb.predict(X_test)

# Accuracy of training and test sets
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Test Accuracy: {:.4}%'.format(test_accuracy * 100))
```

    Training Accuracy: 84.22%
    Test Accuracy: 79.88%


### Evaluate Model
Accuracy, AUC & Confusion Matrix Results


```python
evaluate_model(X_test, y_test, xgb)
```

    Accuracy is :79.88
    
    AUC is :0.76
    
    Confusion Matrix
    ----------------



![png](output_124_1.png)


### Feature Importance Visualization


```python
plot_feature_importances(xgb)
```


![png](output_126_0.png)





    NumberChildrenAtHome         0.170746
    MaritalStatus_S              0.125089
    MaritalStatus_M              0.113394
    Gender_F                     0.081337
    AveMonthSpend                0.053646
    Age                          0.037993
    Gender_M                     0.035457
    Occupation_Skilled Manual    0.033953
    Occupation_Professional      0.029416
    Education_Bachelors          0.025609
    dtype: float32




```python
get_corr(df,xgb)
```


![png](output_127_0.png)



![png](output_127_1.png)



```python

```

# INTERPRET 

Started this project with the goal of finding which customer data features or customer demographics are the best to target for new bike purchases.  We wanted to know how best can we spend our marketing budgets as it relates to the most expensive buyers, which are new customers.  Moreover, the goal is to find the best possible social demographic features to target potential new bike buyers for a series of marketing campaigns over the next budget year. 

Deployed the following machine learning models for the predictions:

1. Decision Tree
2. Random Forest
3. Bagging
4. XGBoost

Of those above models, we evaluated and selected Random Forest results because:

1. Highest predicted buyer who actually bought (69%)
2. Tied for the lowest didn't buy but predicted to buy (83%)

The main Guiding Principal of this project is to find the top data features or those customer demographics to target for marketing campaigns to potential new buyers.  

- **Attributes to Target**
    1. Number of children (more = better)
    2. Martial status (singles)
    3. Gender (males)
    4. Average monthly spend (more)
    5. Age (focus on younger)
    6. Education (some college or higher)


  


```python

```
