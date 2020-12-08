import re
import sqlalchemy
import csv
import requests
import numpy as np
import pandas as pd
import sys
import json
import logging
import sqlalchemy
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions import bif
from iotfunctions.metadata import EntityType
from scripts.simple_mfg_entities import Turbines
from iotfunctions.db import Database
import datetime as dt
from iotfunctions.base import BaseTransformer
#from iotfunctions.bif import EntityDataGenerator
#from ai import settings
from iotfunctions.pipeline import JobController
from iotfunctions.enginelog import EngineLogging
from iotfunctions import system_function
EngineLogging.configure_console_logging(logging.DEBUG)
logger = logging.getLogger(__name__)

import argparse

# from iotfunctions.db import http_request


logging.debug("start")

parser = argparse.ArgumentParser(description='Provide entity and csv information.')
parser.add_argument('-n','--entity_name', dest='entity_type_name',
                    help='Name of entity type')
parser.add_argument('-t','--asset_tags', dest='asset_tags_file',
                    help='Path to tags file')
parser.add_argument('-d', '--data', dest='asset_series_data_file',
                    help='Path to data csv file')
parser.add_argument('-c' ,'--credentials', dest='credentials_path',
                    help='Path to credentials')

parser.add_argument('-f' ,'--dateformat', dest='date_format',
                    help="Optional: Format of datapoint timestamp in strptime") #..\n ex. 2020-05-03 10:30:19.123 => '%Y-%m-%d %H:%M:%S.%f' Format will be inferred otherwise")
parser.add_argument('-co' ,'--datecolumn', dest='date_column',
                    help='Optional: Path to credentials')

parser.add_argument('-fn' ,'--fill_null', action='store_true', dest='fill_null',
                    help='Fill in null values')


args = parser.parse_args()
entity_type_name = args.entity_type_name
credentials_path = args.credentials_path
asset_tags_file = args.asset_tags_file
asset_series_data_file = args.asset_series_data_file
date_format = args.date_format
date_column = args.date_column
if args.fill_null:
    fill_null = True
else:
    fill_null = False

if None in [entity_type_name, credentials_path, asset_tags_file, asset_series_data_file]:
    print("missing args, please ensure to provide values for entity_type_name, credentials_path, asset_tags_file, asset_series_data_file")
    exit()

# args['credentials_path']
# print(args)
# python3.7 scripts/create_entities_using_csv.py -n kb_stork_0915_2 -t ~/Downloads/TagNameEntityDefinition-Engine.csv -d ~/Downloads/CAT_SB.csv -c ~/projects/turbine-demo/credentials/monitor-demo-new.json
'''
if (len(sys.argv) > 0):
    entity_type_name = sys.argv[1]
    asset_tags_file = sys.argv[2]
    asset_series_data_file = sys.argv[3]
    credentials_path = sys.argv[4]
    logging.debug("entity_name %s" % entity_type_name)
else:
    logging.debug("Please provide path to csv file as script argument")
    exit()
'''

'''
# Replace with a credentials dictionary or provide a credentials
# Explore > Usage > Watson IOT Platform Analytics > Copy to clipboard
# Past contents in a json file.
'''
logging.debug("Read credentials")
# with open(credentials_path, encoding='utf-8') as F:
with open(credentials_path) as F:
    credentials = json.loads(F.read())

methods = {'count': 'Count', 'std': 'Std', 'product': 'Product', 'last': 'Last', 'min': 'Minimum', 'max': 'Maximum',
           'sum': 'Sum', 'median': 'Median', 'var': 'Var', 'first': 'First', 'count_distinct': 'Count_distinct', 'mean': 'Mean'}
functions = []
rest_functions = []

print("Reading Tags CSV")

# standardize metrics
tags = pd.read_csv(asset_tags_file)
clean = lambda x: {x: ''.join(re.findall(r'[a-zA-Z-_\d\s:]', x)).lower().strip().replace(' ', '_' )}
updated_names_list = list(map(clean, tags["Metric"]))

timestamp_columns = []


# handle_metric()
# handle_dimension()
# handle_function()
# handle_location()
with open(asset_tags_file, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    point_dimension_values = {
        "label": "",
        "units": "",
        "parameter_name": ""
    }
    metrics = []
    dims = []
    constants = []
    funs = []
    dimension_columns = []
    dimension_column_names = []
    for row in csv_reader:
            try:
                print("printing row")
                print(row)
                parameter_value = row["Value"].replace(" ", '_')
                # parameter_name = ''.join(re.findall(r'\w+', parameter_name))
                parameter_name = list(updated_names_list[line_count].values())[0].replace('-', '_')
                line_count += 1
                print(f"{parameter_name} {parameter_value}")
                logging.debug("Name %s" % parameter_name)
                type = row["DataType"]
                logging.debug("Type %s" % type)
                logging.debug("Value %s" % parameter_value)
                if parameter_name == "":
                    break  # No more rows

                # Create metric
                if row["Point_Data_Type"] == "S":
                    print("________________________ Point point_data_type  %s " %
                          row["Point_Data_Type"])
                    print(
                        "________________________ Point db function name  %s " % row["Function"])
                    if 'string' in type.lower():  # string requires length
                        print('string')
                        print(parameter_name)
                        type = "String"
                        metrics.append(
                            Column(parameter_name, getattr(sqlalchemy, type)(50), nullable=True))
                    elif ('timestamp' in type.lower()) or (('datetime' in type.lower())):
                        print('timestamp')
                        print("setting timestamp column " + parameter_name)
                        timestamp_columns.append(parameter_name)
                        metrics.append(Column(parameter_name, getattr(sqlalchemy, 'String')(50), nullable=True ))
                        # metrics.append(Column(parameter_name, DateTime()))
                        print(type)
                        print("timestamp set")
                    else:
                        print('type: ' + type)
                        metrics.append(Column(parameter_name, getattr(sqlalchemy, type)(), nullable=True))


                # '''
                # Create dimension
                if row["Point_Data_Type"] == "D":
                    logging.debug("________________________ Point point_data_type dimension")
                    dim_to_add = {'parameter_name': parameter_name, 'type': type, 'value':parameter_value}
                    dims.append(dim_to_add)
                    for dim in dims:
                        logging.debug("Adding dimension name to entity type %s" %dim['parameter_name'] )
                        logging.debug("Adding metric type to entity type %s" %dim['type'] )
                        unallowed_chars = "!@#$()"
                        for char in unallowed_chars:
                            dim['parameter_name'] = dim['parameter_name'].replace(char, "")
                        dimension_columns.append(Column(dim['parameter_name'], String(50)))
                        dimension_column_names.append(dim['parameter_name'])
                        logging.debug("Adding cleansed dimension name to entity type %s" % dim['parameter_name'])


                # Create Constant
                if row["Point_Data_Type"] == "C":
                    logging.debug("________________________ Point point_data_type constant")
                    constant_to_add = {'parameter_name': parameter_name, 'type': type, 'value':parameter_value}
                    constants.append(constant_to_add)
                # '''

                # Create Function
                if row["Point_Data_Type"] == "F":
                    # TODO, we need to standardize how many args can be passed in. And handle them properly
                    # bif.PythonExpression
                    # bif.PythonExpression(expression='df["temp"]*df["pressure"]', output_name='volume')
                    print("________________________ Point point_data_type  %s " %
                          row["Point_Data_Type"])
                    print("________________________ Point db data_type  %s " %
                          row["DataType"])
                    print(
                        "________________________ Point db function name  %s " % row["Function"])

                    output_name = parameter_name  # row['Point']
                    expression = row['Function']
                    # merge with "Input Argument list of metric names"
                    input_metrics = ''.join(re.findall(
                        r'\w+', row['Input Arg Value'].lower().replace(' ', ''))).split('|')
                    if len(input_metrics[0]) < 1:
                        print("function requires input metrics, skipping")
                        continue
                    source = input_metrics[0]
                    function_name = row['Function']
                    if function_name in methods.keys():
                        function_name = methods[function_name]
                        input = {"source": source}
                        output_name = function_name.lower() + parameter_name
                        payload = {
                            "functionName": function_name,
                            "granularity": "Daily",
                            "input": input,
                            "output": {
                                "name": output_name
                            },
                            "schedule": {},
                            "backtrack": {},
                            "enabled": True
                        }

                        print(f"appending function {payload}")
                        rest_functions.append(payload)
                        # functions.append(f)
                    elif function_name == 'ratio':
                        # expression = "df['%s'].iloc[-1] / df['%s'].iloc[-1]" % (input_metrics[0], input_metrics[1])
                        function_name = "PythonExpression"
                        expression = "df['%s'] / df['%s']" % (
                            input_metrics[0], input_metrics[1])
                        input = {"expression": expression}
                        output_name = entity_type_name.lower() + "ratio"
                        f = bif.PythonExpression(
                            expression=expression, output_name=output_name)
                        functions.append(f)
                        continue
                    elif function_name == 'multiply':
                        # expression = "df['%s'].iloc[-1] / df['%s'].iloc[-1]" % (input_metrics[0], input_metrics[1])
                        function_name = "PythonExpression"
                        expression = "df['%s'] / df['%s']" % (
                            input_metrics[0], input_metrics[1])
                        input = {"expression": expression}
                        output_name = entity_type_name.lower() + "multiply"
                        f = bif.PythonExpression(
                            expression=expression, output_name=output_name)
                        functions.append(f)
                        continue
                    else:
                        function_name = "PythonExpression"
                        expression = input_metrics[0]
                        f = bif.PythonExpression(
                            expression=expression, output_name=output_name)
                        functions.append(f)
                        continue
            except:
                logging.debug("error parsing tags")
                logging.debug(sys.exc_info()[0])  # the exception instance
                exit()

metrics.append(Column('updated_utc', DateTime(), nullable=True))

columns = tuple(metrics)
print(f"columns {columns}")
print("printing rest_functions")
print(rest_functions)
# exit()
'''
Create a database object to access Watson IOT Platform Analytics DB.
'''
db = Database(credentials=credentials)
db_schema = None
# db_schema = 'bluadmin' #  set if you are not using the default

'''
print("Delete existing Entity Type")
db.drop_table(entity_type_name, schema = db_schema)
'''

#print("Unregister EntityType")
####
# Required input args for creating an entity type
# self, name, db, columns=None, constants=None, granularities=None, functions=None,
#                 dimension_columns=None, generate_days=0, generate_entities=None, drop_existing=False, db_schema=None,
#                 description=None, output_items_extended_metadata=None, **kwargs)
# https://github.com/ibm-watson-iot/functions/blob/60002500117c4559ed256cb68204c71d2e62893d/iotfunctions/metadata.py#L2237
###

# entity = Turbines(
#     name='kalonji_turbine_demo_1',
#     table_name="kalonji_turbine_demo_1",
#     db=db,
#     db_schema=db_schema
# )

dim_table_name = entity_type_name + '_dimension'

'''
# Drop previous dimension table
logging.debug(f"Dropping Dim Table {dim_table_name}")
db.drop_table(dim_table_name, schema=db_schema)
'''

logging.debug("Creating Entity Type")


entity = Turbines(
    name=entity_type_name,
    db=db,
    db_schema=db_schema,
    table_name=entity_type_name,
    columns=columns,
    functions=functions,
    dimension_columns=dimension_columns,
    date_column=date_column,
    date_format=date_format,
    description="Equipment Turbines"
    # generate_entities=['RWS79'],
    # asset_tags_file=asset_tags_file,
)


# entity.generate_dimension_data(entities=['RWS79'])

# entity.read_meter_data(input_file=asset_series_data_file)
# exit()


logging.debug("Register EntityType")
entity.register()  # raise_error=True, publish_kpis=True)

# logging.debug("Generating data")
# entity.generate_data(days=0.5)

logging.debug("Publishing functions")
logging.debug(functions)
# entity.publish_kpis()

print(f"columns {columns}")

#logging.debug("Create Calculated Metrics")
# entity.publish_kpis()

# TODO, finish dimensions once we figure out how they should be defined
# https://www.ibm.com/support/knowledgecenter/SSQR84_monitor/iot/analytics/as_add_dimensions_api.html
# https://www.ibm.com/support/knowledgecenter/SSQR84_monitor/iot/analytics/as_add_dimensions_code.html
# df = pd.read_csv(asset_series_data_file)
# for col in dimension_columns:
#     print(col)
#     url = "https://%s/api/kpi/v1/%s/entityType/%s/dimensional" % (
#         credentials['iotp']['asHost'], credentials['tenantId'], entity_type_name)
#     headers = {'Content-Type': "application/json", 'X-Api-Key': credentials['iotp']['apiKey'],
#                'X-Api-Token': credentials['iotp']['apiToken'], 'Cache-Control': "no-cache", }
#     payload = df[col]
#     r = requests.post(url, headers=headers, json=payload)

for payload in rest_functions:
    # entity_type.db.http_request(object_type='function', object_name=name, request='DELETE', payload=payload)
    print("posting payload")
    print(payload)
    url = "https://%s/api/kpi/v1/%s/entityType/%s/kpiFunction" % (
        credentials['iotp']['asHost'], credentials['tenantId'], entity_type_name)
    headers = {'Content-Type': "application/json", 'X-Api-Key': credentials['iotp']['apiKey'],
               'X-Api-Token': credentials['iotp']['apiToken'], 'Cache-Control': "no-cache", }
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        print("function created")
    else:
        print("failure creating function")
        print(r.status_code)
        print(r.text)
    # entity.db.http_request('kpiFunctions', entity_type_name, 'POST', payload)

logging.debug("Load Metrics Data")
entity.read_meter_data(timestamp_columns=timestamp_columns, input_file=asset_series_data_file, date_format=date_format, date_column=date_column, fill_null=fill_null)


# entity.make_dimension(dim_table_name.upper(), Column('ship', String(50)) )
# entity.generate_dimension_data(entities=entity_ids)

# df_dim['devicetype'] = 'vv'
# df_dim.fillna("N/A")
# db.write_frame(df=df_dim, table_name=dim_table_name.upper(), if_exists='replace')
# print(df_dim)
# dim_table_name = self.table_name + '_dimension'
# self.db.write_frame(df=df_dim, table_name=dim_table_name.upper(), if_exists='replace')
# entity.generate_dimension_data(
    # entities=entity_ids
    # data_item_domain={
    #     "devicetype": "vv"
    # }
# )
logging.debug("Entity Ids Registered")

meta = db.get_entity_type(entity_type_name)
jobsettings = {
    # '_production_mode': False,
    '_start_ts_override': dt.datetime.utcnow() - dt.timedelta(days=10),
    # .strftime('%Y-%m-%d %H:%M:%S'),
    '_end_ts_override': (dt.datetime.utcnow() - dt.timedelta(days=1)),
    '_db_schema': db_schema,
    'save_trace_to_file': True}

logging.info('Instantiated create job')
job = JobController(meta, **jobsettings)
job.execute()

# Check to make sure table was created
print("DB Name %s " % entity_type_name)
print("DB Schema %s " % db_schema)
df = db.read_table(table_name=entity_type_name, schema=db_schema)
print(df.head())

# entity.make_dimension(dim_table_name) #, Column('ship', String(50)))
# entity.make_dimension(dim_table_name.upper()) #, *dimension_columns)

# logging.debug(f"Register entity")
# entity.register(raise_error=True)

# logging.debug(f"Execute local pipeline")
# entity.exec_local_pipeline()

logging.debug(f"Making Dimension Table {dim_table_name}")
entity.make_dimension(dim_table_name.upper()) #*dimension_columns)

'''
# Dimensions
# breaking on entity ids as of migration 9/13
exit()
logging.debug("Registering Entity Ids")
entity_ids = entity.entity_ids
logging.debug(entity_ids)
entity.generate_dimension_data(entities=entity_ids)
exit()
logging.debug(f"Dimension {dim_table_name} registered")
df_dim = pd.DataFrame(columns = dimension_column_names)
df_dim['deviceid'] = entity_ids
df_dim['devicetype'] = "N/A"
df_dim.fillna("N/A")
print(df_dim)
logging.debug(f"Writing unmatched")
# db.start_session()
# table = db.get_table(table_name='kb_lun_0831_6_dimension', schema=db_schema)
# for
# db.write_frame(df=df_dim, table_name=dim_table_name.upper(), if_exists='replace')
entity.write_unmatched_members(df_dim)
logging.debug(f"Done")
entity.generate_dimension_data(entities=entity_ids)
# TODO, add dimension calls
#entity.read_meter_data( input_file="None")
# df = db.read_table(table_name='kb_luny', schema=db_schema)
# df1 = db.read_table(table_name='b_luny_docker', schema=db_schema)
# df = db.read_table(table_name='kb_luny_dimension', schema=db_schema)
'''