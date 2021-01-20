import numpy as np
import pandas as pd
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

'''
stacking arrays
vertical stacking and horizontal stacking
'''
a = np.array([(1, 2, 3), (3, 4, 5)])
b = np.array([(6, 7, 8), (9, 10, 11)])

logger.debug('np.hstack')
logger.debug(np.hstack((a, b)))
'''
 [ [1, 2,3,6,7,8)]
    [3,4,5,9,10,11] ]
'''
logger.debug('np.vstack')
logger.debug(np.vstack((a, b)))
'''
[   [1,2,3]
    [3,4,5]
    [6, 7,8]
    [9,10,11] 
'''

'''
matrix math

'''
a = np.array([(1, 2, 3), (3, 4, 5)])
b = np.array([(1, 2, 3), (3, 4, 5)])
logger.debug(a + b)  # [ (2, 4, 6)  , (6, 8, 10) ]
logger.debug(a * b)  # [ (1, 4, 9)  , (9, 16, 25) ]
logger.debug(a / b)  # [ (1, 1, 1)  , (1, 1, 1) ]

'''
Sum of axis
            axis 0
            8,  9,
axis 1      10, 11
            12, 13
'''
b = np.array([(8, 9), (10, 11), (12, 13)])
logger.debug(b.sum(axis=0))  # [ 30 33 ]
logger.debug(b.sum(axis=1))  # [ 17 21 25 ]

# Square root
logger.debug('square root')
logger.debug(np.sqrt(b))

# Standard deviation
logger.debug('standard deviation')
logger.debug(np.std(b))

# Get  min , max and sum
a = np.array([(8, 1, 1, 1), (1, 1, 1, 15)])
logger.debug(a.min())  # 1
logger.debug(a.max())  # 15
logger.debug(a.sum())  # 29

# Get 10 alues from 1 to 3
a = np.linspace(1, 3, 10)
logger.debug(a)
'''
[1.         1.22222222 1.44444444 1.66666667 1.88888889 2.11111111
 2.33333333 2.55555556 2.77777778 3.        ]
'''

'''
Reshape

    8,  9,  10, 11      >>  8 ,9
    12, 13, 14, 15          10, 11
                            12, 13
                            14, 15

Slicing
     8,  9      >>  9 ,11
    10, 11
    12, 13
    14, 15

# Change shape of array
a = np.array( [ (8,9,10,11), (12,13,14,15)]  )
logger.debug('before reshape')
logger.debug(a)
logger.debug('after reshape and before slicing')
a = a.reshape(4,2)
logger.debug(a)

# slicing array
# Get size
# use : to get all rows of data including 0 for first one.   00:   gets all rows
logger.debug('after slicing')
logger.debug(a[0,1])  # 9
logger.debug(a[3,0])  # 14
logger.debug(a[2,1])  # 13
logger.debug(a[0:,1])  #  [9, 11, 13, 15]  Get all the rows when using 0:data from second column
logger.debug(a[0:,0])  #  [8, 10, 12, 14]   Get all the rows when using 0:  from first column
logger.debug(a[0:3,0])  #  [8, 10, 12  ]   Ignore items after the :
logger.debug(a[1:4,1])  #  [11, 13, 15  ]   Ignore items after the :

# Find size of array
a = np.array(  (1,2,3)   )
logger.debug('data size is  ')
logger.debug(a.size)

# Find shape of array
a = np.array( [ (1,2,3, 4, 5), (1,2,3, 4, 5)]   )
logger.debug('data shape is 2, 5 ')
logger.debug(a.shape)


# Find dimension of NP arrays self.assert_
a = np.array(  [ (1,2,3) , (2,3,4) ] )
logger.debug('dimension is 2 ')
logger.debug(a.ndim)

# Find dimensinos of NP arrays self.assert_
a = np.array(  (1,2,3)   )
logger.debug('dimension is 1 ')
logger.debug(a.ndim)

# Find dimensinos of NP arrays self.assert_
a = np.array(  (1,2,3)   )
logger.debug('data type is int64 ')
logger.debug(a.dtype)

# NP arrays are faster
SIZE =  1000
L1 =  range (SIZE)
L2 =  range (SIZE)

A1 =  np.arange(SIZE)
A2 =  np.arange(SIZE)

start = time.time()
result =  [(x,y) for x, y in zip(L1, L2) ]
logger.debug (  ( time.time() - start ) * 1000  )

start = time.time()
result = A1 + A2
logger.debug (  ( time.time() - start ) * 1000  )

# NP arrays are less memory
# Range is an array of  1000 numbers stored in S.
S = range(1000)
logger.debug( sys.getsizeof(5)*len(S) )

D = np.arange( 1000 )
logger.debug(D.size * D.itemsize)

# Single dimensional array
a = np.array ( [1,2,3] )
logger.debug(a)

# 2  dimensional array
logger.debug("2--")
numpy_data = np.array(  [ (1,2,3) , (2,3,4) ]     )
logger.debug(numpy_data)

# Convert np array into a dataframe
logger.debug("3--")
numpy_data = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(data=numpy_data, index=["row1", "row2"], columns=["column1", "column2"])
logger.debug(df)

'''
# Slicing Panda Dataframes
logger.debug("Slicing Panda Dataframes")
df = pd.DataFrame(np.arange(20).reshape(5, 4), columns=["A", "B", "C", "D"])
logger.debug(df)

# Select a Panda Dataframe column
logger.debug(df.loc[:, "A"])
logger.debug(df["A"])
logger.debug(df.A)

# To Select multiple Panda Dataframe column
logger.debug(df.loc[:, ["A", "C"]])
logger.debug(df[["A", "C"]])

#  To select a row in a Pandas Dataframe  by its label
logger.debug(df.loc[1])

#  To select multiple rows in a Pandas Dataframe  by its label
logger.debug(df.loc[[0, 1]])

# Accessing values by row and column label.
logger.debug(df.loc[0, "D"])

# Accessing values in row for multiple column label.
logger.debug(df.loc[1, ["A", "C"]])

# Accessing values from multiple rows but same column.
logger.debug(df.loc[[0, 1], "B"])

# You can select data from a Pandas DataFrame by its location. Note, Pandas indexing starts from zero.
# Select a row by index location.
logger.debug(df.iloc[0])

# Select data at the specified row and column location, index starts with 0.  format row, col
logger.debug(df.iloc[0, 3])

# Select list of rows and columns
logger.debug(df.iloc[[1, 2], [0, 1]])

# Slicing Rows and Columns using labels
# You can select a range of rows or columns using labels or by position.
# To slice by labels you use loc attribute of the DataFrame.

# Slice rows by label.  [ rows:rows, columns:columns]
logger.debug(df.loc[1:3, :])

# Slice by columns by label.  [ rows:rows, columns:columns]
logger.debug(df.loc[:, "B":"D"])

# Slicing Rows and Columns by position
# To slice a Pandas dataframe by position use the iloc attribute. Remember index starts from 0 to (number of rows/columns - 1).
logger.debug(df.iloc[0:2, :])

# To slice columns by index position.
logger.debug(df.iloc[:, 1:3])

# To slice row and columns by index position.
#       df.iloc[row_start : row_finish, col_start:col_finish] )
logger.debug(df.iloc[1:2, 1:3])
logger.debug(df.iloc[:2, :2])

#  Subsetting by boolean conditions
# You can use boolean conditions to obtain a subset of the data from the DataFrame.
logger.debug(df[df.B == 9])
logger.debug(df.loc[df.B == 9])

#  Return rows  in  with column B that have a 9 or 13
logger.debug(df[df.B.isin([9, 13])])

# Rows that match multiple boolean conditions.
logger.debug(df[(df.B == 5) | (df.C == 10)])

# Select rows whose column does not contain the specified values.
logger.debug(df[~df.B.isin([9, 13])])

# Select columns based on row value
# To select columns whose rows contain the specified value.
logger.debug(df.loc[:, df.isin([9, 12]).any()])

# Subsetting using filter method
# Subsets can be created using the filter method like below.
logger.debug(df.filter(items=["A", "D"]))

# Subsets of a rowu using index.
logger.debug(df.filter(like="2", axis=0))
# Subsets of a row using reg expression not  columns AB .
logger.debug(df.filter(regex="[^AB]"))

# Indexing using column names
'''
In [203]: df = pd.DataFrame(np.random.randint(n / 2, size=(n, 2)), columns=list('bc'))

In [204]: df.index.name = 'a'

In [205]: df
Out[205]: 
   b  c
a      
0  0  4
1  0  1
2  3  4
3  4  3
4  1  4
5  0  3
6  0  1
7  3  4
8  2  3
9  1  1

In [206]: df.query('a < b and b < c')
Out[206]: 
a  b  c      
2  3  4
'''


def f(df, parameters=None):
    import numpy as np
    return np.where(df['run_status'] == 5, "Online", "Offline")


def m(df, parameters=None):
    # Converts a 0 1 integer array into a string value in a dataframe
    import numpy as np
    # = np.where(df== 0, "Online", "Offline")
    logger.debug("fun m")
    logger.debug(df['scheduled_maintenance'])
    logger.debug(df['unscheduled_maintenance'])
    return np.where((df['unscheduled_maintenance'] == 1) | (df['scheduled_maintenance'] == 1), "Being Serviced",
                    "Not being serviced")


def l(df, parameters=None):
    df = pd.DataFrame({'Age': [30, 20, 22, 40, 32, 28, 39],
                       'Color': ['Blue', 'Green', 'Red', 'White', 'Gray', 'Black',
                                 'Red'],
                       'Food': ['Steak', 'Lamb', 'Mango', 'Apple', 'Cheese',
                                'Melon', 'Beans'],
                       'Height': [165, 70, 120, 80, 180, 172, 150],
                       'Score': [4.6, 8.3, 9.0, 3.3, 1.8, 9.5, 2.2],
                       'State': ['NY', 'TX', 'FL', 'AL', 'AK', 'TX', 'TX']
                       },
                      index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Dean',
                             'Christina', 'Cornelia'])

    logger.debug("\n -- loc -- \n")
    logger.debug(df.loc[df['Age'] < 30, ['Color', 'Height']])

    logger.debug("\n -- iloc -- \n")
    logger.debug(df.iloc[(df['Age'] < 30).values, [1, 3]])

    return


def score(df, parameters=None):
    left = pd.DataFrame({'evt_timestamp': [pd.Timestamp('2020-04-10 07:46:14.687196'),
                                           pd.Timestamp('2020-04-10 07:41:14.687196'),
                                           pd.Timestamp('2020-04-10 07:36:14.687196'),
                                           pd.Timestamp('2020-04-05 21:37:04.209610'),
                                           pd.Timestamp('2020-04-05 21:42:09.209610'),
                                           pd.Timestamp('2020-04-05 21:47:00.209610'),
                                           pd.Timestamp('2020-04-10 07:31:14.687196'),
                                           pd.Timestamp('2020-04-10 07:26:14.687196')],
                         'drvn_p1': [19.975879, 117.630665, 17.929952, 1.307068,
                                     0.653883, 0.701709, 16.500000, 16.001709],
                         'deviceid': ['73001', '73001', '73001', '73000',
                                      '73000', '73000', '73001', '73001'],
                         'drvr_rpm': [165, 999, 163, 30,
                                      31, 33, 150, 149],
                         'anomaly_score': [0, 0, 0, 0,
                                           0, 0, 0, 0]
                         },
                        index=[0, 1, 2, 3,
                               4, 5, 6, 7])

    right_a = pd.DataFrame({'anomaly_score': [1, -1, 1, 1, 1],
                            'drvn_p1': [19.975879, 117.630665, 17.929952, 16.500000, 16.001709],
                            'drvr_rpm': [165, 999, 163, 150, 149]
                            },
                           index=[0, 1, 2, 6, 7])

    # 'asset_id': ['73001', '73001', '73001', '73001']
    logger.debug('left.dtypes')
    logger.debug(left.dtypes)
    logger.debug("\n -- Left starts with   -- \n")
    logger.debug(left)

    # Using DataFrame.insert() to add a column
    # left.insert(2, "Age", [21, 23, 24, 25,21, 23, 24, 25], True)
    # logger.debug('left.added age')
    # logger.debug(left)

    logger.debug("\n -- Right starts with   -- \n")
    logger.debug('right_a.dtypes')
    logger.debug(right_a.dtypes)
    logger.debug(right_a)

    # Add asset_id to the right_a df
    right_a['deviceid'] = '73001'
    # Get the original evt_timestamp from the original df and insert it to the scored df so that you avoid column mismatch when merging.
    right_a.insert(1, 'evt_timestamp', left.loc[left['deviceid'] == '73001', ['evt_timestamp']], True)
    logger.debug("\n -- Right after adding deviceid column and evt_timestamp  -- \n")
    logger.debug(right_a)

    # Set Pandas display options
    # https://towardsdatascience.com/how-to-show-all-columns-rows-of-a-pandas-dataframe-c49d4507fcf
    pd.set_option('display.max_columns', None)
    # pd.reset_option(“max_columns”)
    pd.set_option('max_rows', None)
    # logger.debug(right_a)

    # Concat the rows of two panda data frames
    # https://www.datacamp.com/community/tutorials/joining-dataframes-pandas
    # df_rows = pd.concat([left, right_a], ignore_index=False)
    # logger.debug('df_rows')
    # logger.debug(df_rows)

    #  Concat the columns of two panda data frames
    # https://www.datacamp.com/community/tutorials/joining-dataframes-pandas
    # df_concat_cols = pd.concat([left, right_a], axis=1)
    # logger.debug('df_concat_cols')
    # logger.debug(df_concat_cols)

    # Merge the columns of two panda data frames
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
    # https://www.datacamp.com/community/tutorials/joining-dataframes-pandas
    # df_merg_cols = pd.merge(right_a, left, on='asset_id' )
    # logger.debug('df_cols merge')
    # logger.debug(df_merg_cols)

    #  Merge the columns of two panda data frames
    #  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
    # df_merge_left = left.merge(right_a, how='left', on=['anomaly_score','asset_id','drvr_rpm','drvn_p1'])
    # logger.debug('df_merge_left')
    # logger.debug(df_merge_left)

    #  Merge the columns of two panda data frames
    #  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
    # df_merge_right = right_a.merge(right_a, how='right', on=['anomaly_score','asset_id','drvr_rpm','drvn_p1'])
    # logger.debug('df_merge_right')
    # logger.debug(df_merge_right)

    # how to convert to mintes
    # s = pd.to_timedelta(s_df['evt_timestamp'])
    # s_df['evt_timestamp'] = s / pd.offsets.Minute(1)

    # logger.debug("Converting a column to float")
    # logger.debug ( s )
    # s_df['evt_timestamp'] = s_df['evt_timestamp'].dt.strftime('%B %d, %Y, %r').astype(float)
    # s_df['evt_timestamp'] = s_df['evt_timestamp'].dt.strftime('%B %d, %Y, %r').astype(float)
    # s_df['evt_timestamp'] = pd.to_datetime(s_df['evt_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f000Z')
    # s_df['evt_timestamp'] = pd.to_datetime(df['evt_timestamp'], format='%Y-%m-%dT%H:%M:%S.%f000Z')

    # s_df['evt_timestamp'] = datetime.datetime.strptime( s_df['evt_timestamp'], '%Y-%m-%s %h:%m:%s').isoformat()

    #  Combine the columns of two panda data frames
    df_combine_first = right_a.combine_first(left)
    logger.debug('df_combine_first')
    logger.debug(df_combine_first)

    # Validate you are getting good data for a single asset by plotting data
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(r'/Users/carlos.ferreira1ibm.com/ws/isolation-forestCharts.pdf') as export_pdf:
        # df_combine_first.loc[df_combine_first['deviceid'] == '73001']['evt_timestamp'].dt.strftime('%Y-%m-%d')
        # plt.plot(df_combine_first.loc[df_combine_first['deviceid'] == '73001']['drvn_p1'], color='green', label='drvn_p1')
        # plt.plot(df_combine_first.loc[df_combine_first['deviceid'] == '73001']['anomaly_score'] * 10, color='red', label='anomaly_score')

        # Create figure and plot space
        fig, ax = plt.subplots(figsize=(12, 12))

        # Add x-axis and y-axis use ax.plot for line and ax.bar for bar chart
        logger.debug(df[(df.B == 5) & (df['evt_timestamp'] < pd.Timestamp('2020-04-10 07:41:14.687196'))])

        ax.plot(df_combine_first.loc[df_combine_first['deviceid'] == '73001'].sort_values(by='evt_timestamp',
                                                                                          ascending=True)[
                    'evt_timestamp'],
                df_combine_first.loc[df_combine_first['deviceid'] == '73001'].sort_values(by='evt_timestamp',
                                                                                          ascending=True)['drvn_p1'],
                color='purple')

        # Set title and labels for axes
        ax.set(xlabel="Date",
               ylabel="drvn_p1 (psi)",
               title="Pressure\n Test")

        # plt.show()
        export_pdf.savefig()
        plt.close()

    # Test values for x axis labels
    logger.debug("Test values for x axis labels")
    logger.debug(df_combine_first.loc[df_combine_first['deviceid'] == '73001']['evt_timestamp'])

    orig_left = left

    logger.debug("\n -- simple merge right  -- \n")
    # left['anomaly_score'] = left['anomaly_score'].astype(int)
    right_a['anomaly_score'] = right_a['anomaly_score'].astype(int)
    left['anomaly_score'] = left['anomaly_score'].astype(int)
    # left['evt_timestamp'] = left['evt_timestamp'].astype(datetime)
    left = left.merge(right_a, how="left")
    logger.debug(left)

    logger.debug("\n -- simple merge left  -- \n")
    right_a['anomaly_score'] = right_a['anomaly_score'].astype(int)
    orig_left['anomaly_score'] = orig_left['anomaly_score'].astype(int)

    orig_left = orig_left.merge(right_a, how="right")
    logger.debug(orig_left)

    logger.debug("\n -- before  -- \n")
    logger.debug(left.loc[left['deviceid'] == '73001', ['deviceid', 'evt_timestamp', 'drvn_p1', 'drvr_rpm']])
    logger.debug(left.loc[left['deviceid'] == '73000', ['deviceid', 'evt_timestamp', 'drvn_p1', 'drvr_rpm']])
    '''

    logger.debug("\n -- after merged_asof left , right -- \n")
    merged_asof = pd.merge_asof(left, right_a,
                                by = 'asset_id')
    #logger.debug(merged_asof.loc[merged_asof['asset_id'] == '73001', ['asset_id', 'evt_timestamp', 'drvn_p1', 'drvr_rpm', 'anomaly_score']])
    #logger.debug(merged_asof.loc[merged_asof['asset_id'] == '73000', ['asset_id', 'evt_timestamp', 'drvn_p1', 'drvr_rpm', 'anomaly_score']])
    logger.debug(merged_asof)
    '''
    return


def isolation(parameters=None):
    # Import csv file
    df = pd.read_csv("/Users/carlos.ferreira1ibm.com/ws/shell/data/turbine_data.csv", header=None)
    logger.debug(df.head())
    from sklearn.ensemble import IsolationForest
    X = [[-1.1], [0.3], [0.5], [100]]
    logger.debug(X)
    clf = IsolationForest(random_state=0).fit(X)
    clf.predict([[0.1], [0], [90]])
    np.array([1, 1, -1])
    return

def auto_ai(df, parameters=None):

    import numpy as np
    import requests
    import json

    data_array = df.to_numpy()
    data_list = data_array.tolist()
    #result = np.zeros(data_array.shape[0])
    #result[:] = np.NaN
    token = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImRhdHRhcmFvIiwicm9sZSI6IlVzZXIiLCJwZXJtaXNzaW9ucyI6WyJhY2Nlc3NfY2F0YWxvZyIsImNhbl9wcm92aXNpb24iLCJzaWduX2luX29ubHkiXSwic3ViIjoiZGF0dGFyYW8iLCJpc3MiOiJLTk9YU1NPIiwiYXVkIjoiRFNYIiwidWlkIjoiMTAwMDMzMTAwMiIsImF1dGhlbnRpY2F0b3IiOiJkZWZhdWx0IiwiaWF0IjoxNjA1MDAwODEyLCJleHAiOjE2MDUwNDM5NzZ9.Dmk1cFnAdmz3bVTpx0v3x7HdNJ4YHlFc4YRXF8R-qko3kRBJmRnh3qDFBbont7WlByAmMg2oY92ef7KKe9tic7BSJ9kYGwScXdregwZsvsBXVqhgXQ4IMvoNVQfM4y6m7UVvBg4ZqF_nRFWK3DuLz4p_SGcxkpcaHi7vbl5iKBSVDc7UgRV3JTLwjpAuJuRUTZVtisBYshC1oUdLLTDMyImasDbasjucODquCaL6ZZdEOM2A1I5zZH6TBWS6qcHVtN0crd7TbQsQg-S7ohtR01EPnXdu-APXp0sTQCTNq7Eb3izysVkp9QJ4GMamx69bnSrpgrDNawIyB5KVva9fkg'
    str1 = 'Bearer ' + token
    logger.debug(str1)
    header = {'Content-Type': 'application/json', 'Authorization': str1}

    payload_scoring = {"input_data": [{"fields": ["INV1A - Module 2 AC Power Smoothed"], "values": data_list}]}
    logger.debug(
        "\npayload_scoring: " + str(type(payload_scoring)) + "\ndata_list: " + str(type(data_list)) + "\ndata_array" + str(type(
            data_array)))
    response_scoring = requests.post(
        'https://zen-cpd-zen.maximodev-14cc747866faab74c9640c8ac367af7f-0000.us-south.containers.appdomain.cloud/v4/deployments/4887e338-4718-4d6c-98f4-973cad74954f/predictions',
        json=payload_scoring, headers=header)

    logger.debug("Scoring response")
    logger.debug(response_scoring.status_code)
    data_scores = json.loads(response_scoring.text)
    logger.debug(data_scores)

    x = 0
    for value in np.array(data_scores['predictions'][0]['values']):
        if value == "NULL":
            logger.debug('Null was found')
            data_scores['predictions'][0]['values'][x] = np.NaN
        x = x + 1

    #result = None
    #try:
    #    result = np.array(data_scores['predictions'][0]['values'])
    #except Exception as e:
    #    logger.error('Exception ' + str(e))
    #    data_array[:] = np.NaN
    #    pass
    #return result

    result =  {'predictions': [{'fields': ['$L-MET1 - Average Temperature Corrected POA Irradiance Smoothed'], 'values': [[-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807], [-2.30374275492807],  [926.7670532161372], [926.501495688029]]}]}

    return np.array(data_scores['predictions'][0]['values'])


def get_status(df, state_column, state_name, state_metric_name, parameters=None):
    # Not used
    '''
    d = {'evt_timestamp': [pd.Timestamp('2020-04-10 07:26'),
                       pd.Timestamp('2020-04-10 07:31'),
                       pd.Timestamp('2020-04-10 07:46'),
                       pd.Timestamp('2020-04-10 07:51'),
                       pd.Timestamp('2020-04-05 21:32'),
                       pd.Timestamp('2020-04-05 21:37'),
                       pd.Timestamp('2020-04-05 21:42'),
                       pd.Timestamp('2020-04-10 07:51'),
                       pd.Timestamp('2020-04-10 08:00')],
     'deviceid': ['73001', '73001', '73001', '73001',
                  '73000', '73000', '73000', '73001', '73001'],
     'runningstatus': ["RUNNING", "STOPPED", "RUNNING", "RUNNING",
                       "STOPPED", "RUNNING", "RUNNING", "STOPPED", "RUNNING"]
     }
    '''
    #df = pd.DataFrame(d, index=pd.Series(['73001', '73001', '73001', '73000',
    #                                              '73000', '73000', '73000', '73001', '73001'], name='Name'))

    df = pd.DataFrame({'running_status': ['RUNNING', 'RUNNING', 'RUNNING', 'STOPPED',
                                          'STOPPED', 'STOPPED', 'RUNNING', 'STOPPED',
                                          'RUNNING', 'RUNNING']},
                        index=[('73001', pd.Timestamp('2020-04-10 07:31')), ('73001', pd.Timestamp('2020-04-10 07:46')), ('73001', pd.Timestamp('2020-04-10 07:51')), ('73001', pd.Timestamp('2020-04-10 08:01')),
                               ('73000', pd.Timestamp('2020-04-05 21:37')), ('73000', pd.Timestamp('2020-04-05 21:42')),('73000', pd.Timestamp('2020-04-05 21:47')),('73000', pd.Timestamp('2020-04-05 21:52')),
                               ('73001', pd.Timestamp('2020-04-10 08:06')), ('73001', pd.Timestamp('2020-04-10 08:11'))])
    print(df)
    df.index = pd.MultiIndex.from_tuples(df.index, names=['id', 'evt_timestamp'])
    print(df)

    logger.debug("Original Simulation Data to test downtime calculations")
    logger.debug("start df ----- %s " % df)
    logger.debug("start df columns ----- %s " % df.columns)
    logger.debug("start state_column  ----- %s " %state_column)
    logger.debug("start logger.debug state_name %s " %state_name)
    logger.debug("start print state_metric_name %s " %state_metric_name)

    logger.debug(df.index.names)
    # List unique values in the df['name'] column
    logger.debug('List of Running Status')
    states = df[state_column].unique()
    logger.debug(states)

    logger.debug("Original Simulation Data looking at rows")
    for index, row in df.iterrows():
        logger.debug("original rows")
        logger.debug(row)

    # Initialize status you need to find running times for
    pd.set_option('display.max_columns', None)
    for state in states:
        df[state_name] = 0
        df[state] = 0

    logger.debug("Debugging")
    entity_index_name = df.index.names[0]
    time_index_name = df.index.names[1]
    df.reset_index(inplace=True)
    #logger.debug('Here are entity_index_name, time_index_name, df.columns ', entity_index_name, time_index_name,
    #             df.columns)
    asset_list = df[entity_index_name].unique().tolist()
    logger.debug("List of unique equipment")
    logger.debug(asset_list)

    logger.debug("Analyze Index")
    for asset in asset_list:
        logger.debug("Get rows just for single asset %s --" % asset)
        df_asset = df.loc[df[entity_index_name] == asset]
        logger.debug("Rows just for %s" % asset)
        logger.debug(df_asset)

        # rows = [list(r) for i, r in df_asset.iterrows()]
        first_row = True
        # self.state_column_idx = list(df_asset.columns).index(self.state_column)

        for index, row in df_asset.iterrows():
            if first_row == False:
                logger.debug("iterate rows")
                #logger.debug(row['evt_timestamp'], row[df[entity_index_name]], row[state_column])

                # Calculate mins running
                mins_running = row['evt_timestamp'] - laststatus_timestamp
                laststatus_timestamp = row['evt_timestamp']
                logger.debug("New status_timestamp %s " % laststatus_timestamp)
                logger.debug("mins_running %s " % mins_running)
                mins = mins_running.total_seconds() / 60
                logger.debug("mins are %s " % mins)
                logger.debug("self.state_column is  %s " %state_column)
                # Update original dataframe with calculated minutes running
                df.loc[
                    (df[entity_index_name] == asset) & (
                            df['evt_timestamp'] == row['evt_timestamp']), [
                        row[state_column]]] = mins_running.total_seconds() / 60

                # df.loc[(df['deviceid'] == asset) & (df['evt_timestamp'] == row['evt_timestamp'], df[state_name]  = mins_running.total_seconds() / 60
            else:
                logger.debug("First Row")
                #logger.debug(row['evt_timestamp'], row[df[entity_index_name]], row[state_column])
                first_row = False
                laststatus_timestamp = row['evt_timestamp']
            logger.debug("Previous status_timestamp %s " % laststatus_timestamp)

        for item in states:
            logger.debug("\n -- %s Device total mins running in state %s -- \n" % (asset, item))
            logger.debug(df.loc[df[entity_index_name] == asset, item].sum())
            logger.debug("\n ---- \n")

    # logger.debug("\n -- iloc -- \n")
    # logger.debug(df.iloc[(df['Age'] < 30).values, [1, 3]])

    logger.debug('Finished State Calculations ')
    for asset in asset_list:
        logger.debug("Get rows just for single asset %s --" % asset)
        df_asset = df.loc[df[entity_index_name] == asset]
        logger.debug("Rows just for %s" % asset)
        logger.debug(df_asset)

    # Reset DF the index back to what it was
    df.set_index([entity_index_name, time_index_name], inplace=True)

    # Assign state_name with minutes from state they want
    df[state_metric_name] = df[state_name]
    logger.debug('df[state_metric_name]')
    logger.debug(df[state_metric_name])

    logger.debug('Final entire DF we are returning with state_name and minutes |||  ')
    logger.debug(df)
    return df

    '''
    last_status_timestamp = 1
    last_status = off

    # Check if you are comi=--g off downtime  and calculate event_time_down
    If
    running_status = on and last_status = off
    event_time_down = current_time - laststatus_timestamp_changed
    laststatus = running_status
    laststatus_timestamp_changed = rs_timestamp

    # Check if you are starting new off downtime
    if runningstatus = off and laststatus = on
    event_time_up = current_time - laststatus_timestamp_changed
    laststatus = running_status
    laststatus_timestamp_changed = rs_timestamp
    '''


def main(args):
    inputfile = args
    logger.debug('Reading args %s', inputfile)

    '''
    # Metrics WORKDS

    df = pd.DataFrame({'Value': ['RUNNING', 'RUNNING', 'STOPPED', 'RUNNING', 'STOPPED']},
                        index=[('73000A', pd.Timestamp('2020-05-04 21:27:00')), ('73000A', pd.Timestamp('2020-05-04 21:32:00')), ('73000A', pd.Timestamp('2020-05-04 21:37:00')), ('73001A', pd.Timestamp('2020-10-04 07:26:00')),('73001A', pd.Timestamp('2020-10-04 07:26:00'))])
    print(df)
    df.index = pd.MultiIndex.from_tuples(df.index, names=['id', 'evt_timestamp'])
    print(df)

    pd.set_option('display.max_columns', None)
    arrays = [np.array(['73000A', '73001A', '73000A', '73001A' ]),
              np.array(['2020-05-04 21:27:00', '2020-05-04 21:32:00', '2020-10-04 07:26:00', '2020-10-04 07:31:00'])]
    #np.array(
    #    [pd.Timestamp('2020-05-04 21:27:00'), pd.Timestamp('2020-05-04 21:32:00'), pd.Timestamp('2020-10-04 07:26:00'),
    #     pd.Timestamp('2020-10-04 07:31:00'])]

    s = pd.Series(np.random.randn(4), index=arrays)
    logger.debug('s')
    logger.debug(s)
    df = pd.DataFrame(np.random.randn(4, 4), index=arrays)
    logger.debug('df')
    logger.debug(df)
    logger.debug("start df ----- %s " % df.columns)

    df.columns = index
    logger.debug('df')
    logger.debug(df)
    # Metrics WORKS ENDS
    '''

    '''
    # Metrics
    # run_status = 5 running  0 not running
	# scheduled_maintenance = 1 maintenance or 0 None
	# unscheduled_maintenance = 0 None or 1 unscheduled maintenance


    #  Create numpy dimensional array of run status single metric modes running ie 1  versus not running ie 0
    numpy_data = np.array([[0], [1], [0], [5], [0]])
    logger.debug("numpy_data  for f -----------------------------------")
    logger.debug(numpy_data)
    # Convert np array into a dataframe
    df = pd.DataFrame(data=numpy_data, index=["row1", "row2", "row3", "row4", "row5"], columns=["run_status"])
    logger.debug(df)

    val = f(df=df, parameters=None)
    logger.debug("Status returned as DF")
    logger.debug(val)

    #  Create numpy dimensional array of scheduuled_maintenance and scheduuled_maintenance metric
    #  Modes running ie 1  versus not running ie 0
    numpy_data = np.array([[0,0], [1,0], [0,0], [0,0], [0,0]])
    logger.debug("numpy_data  for maintenance -----------------------------------")
    logger.debug(numpy_data)
    # Convert np array into a dataframe
    df = pd.DataFrame(data=numpy_data, index=["row1", "row2", "row3", "row4", "row5"], columns=["scheduled_maintenance", "unscheduled_maintenance" ])
    logger.debug(df)
    val = m(df=df, parameters=None)
    logger.debug("Status returned as DF")
    logger.debug(val)


    #  Create numpy dimensional array of rpm and pressure to  separate df for scoring then merge the two DFs
    val = score(df=df, parameters=None)
    logger.debug("Status returned as DF for location test")
    logger.debug(val)
    logger.debug ( 'Done testing ' )
    '''

    #  Check for how long an asset is running in a specific state
    logger.debug("Calculate downtime returned as DF for location test")
    val = get_status(df=df, state_column="running_status",  state_name="RUNNING", state_metric_name="running_minutes", parameters=None)
    logger.debug('Done testing ')

    #  Create numpy dimensional array of rpm and pressure to  separate df for scoring then merge the two DFs
    # val = isolation( parameters=None)
    # logger.debug ( 'Done testing Isolation Forest ' )

    #  Check for how long an asset is running in a specific state
    #logger.debug("Calculate auto_AI regression")
    # Import csv file
    #asset_series_data_file = "/Users/carlos.ferreira1ibm.com/Downloads/turbine-demo-csv/data/INV1A_data_m1t_smooth.csv"
    #df = pd.read_csv(asset_series_data_file)
    #df = pd.read_csv("/Users/carlos.ferreira1ibm.com/Downloads/turbine-demo-csv/data", header=None)
    #logger.debug(df.head())
    #val = auto_ai(df=df, parameters=None)
    #logger.debug('Done testing ')



if __name__ == "__main__":
    logger.debug("here")
    main(sys.argv[1:])




    '''
    # Example Custom Function
    def f(df, parameters=None):
    import numpy as np
    import requests
    import json

    data_array = df.to_numpy()
    data_list = data_array.tolist()
    token = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImRhdHRhcmFvIiwicm9sZSI6IlVzZXIiLCJwZXJtaXNzaW9ucyI6WyJhY2Nlc3NfY2F0YWxvZyIsImNhbl9wcm92aXNpb24iLCJzaWduX2luX29ubHkiXSwic3ViIjoiZGF0dGFyYW8iLCJpc3MiOiJLTk9YU1NPIiwiYXVkIjoiRFNYIiwidWlkIjoiMTAwMDMzMTAwMiIsImF1dGhlbnRpY2F0b3IiOiJkZWZhdWx0IiwiaWF0IjoxNjA1MDAwODEyLCJleHAiOjE2MDUwNDM5NzZ9.Dmk1cFnAdmz3bVTpx0v3x7HdNJ4YHlFc4YRXF8R-qko3kRBJmRnh3qDFBbont7WlByAmMg2oY92ef7KKe9tic7BSJ9kYGwScXdregwZsvsBXVqhgXQ4IMvoNVQfM4y6m7UVvBg4ZqF_nRFWK3DuLz4p_SGcxkpcaHi7vbl5iKBSVDc7UgRV3JTLwjpAuJuRUTZVtisBYshC1oUdLLTDMyImasDbasjucODquCaL6ZZdEOM2A1I5zZH6TBWS6qcHVtN0crd7TbQsQg-S7ohtR01EPnXdu-APXp0sTQCTNq7Eb3izysVkp9QJ4GMamx69bnSrpgrDNawIyB5KVva9fkg'
    str1 = 'Bearer ' + token
    logger.debug(str1)
    header = {'Content-Type': 'application/json', 'Authorization': str1}

    payload_scoring = {"input_data": [{"fields": ["INV1A - Module 2 AC Power Smoothed"], "values": data_list}]}
    logger.debug(
        "\npayload_scoring: " + str(type(payload_scoring)) + "\ndata_list: " + str(type(data_list)) + "\ndata_array" + str(type(
            data_array)))
    response_scoring = requests.post(
        'https://zen-cpd-zen.maximodev-14cc747866faab74c9640c8ac367af7f-0000.us-south.containers.appdomain.cloud/v4/deployments/4887e338-4718-4d6c-98f4-973cad74954f/predictions',
        json=payload_scoring, headers=header)

    logger.debug("Scoring response")
    logger.debug(response_scoring.status_code)
    data_scores = json.loads(response_scoring.text)
    logger.debug(data_scores)
    return np.array(data_scores['predictions'][0]['values'])
    '''