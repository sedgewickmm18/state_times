import json
import logging
import datetime as dt
from iotfunctions.db import Database
from iotfunctions.enginelog import EngineLogging
from poc.functions import State_Timer
import pandas as pd
from scripts.test_entities import Equipment
from iotfunctions.pipeline import JobController

logger = logging.getLogger(__name__)

with open('credentials_Monitor-Demo2.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())
db_schema = 'bluadmin'
db = Database(credentials=credentials)
entity_type_name = 'Container5'
entityType = entity_type_name
# Use create_entities_usingcsv.py to create Entity Type.
'''
#db.drop_table(entity_type_name, schema = db_schema)
entity = Equipment(name = entity_type_name,
                db = db,
                db_schema = db_schema,
                description = "Smart Connect Operations Control Center",
                )

entity.register(raise_error=False)
'''
#Register function so that you can see it in the UI
#db.unregister_functions(["State_Timer"])
#db.register_functions([State_TimerV2])
'''
meta = db.get_entity_type(entityType)
jobsettings = {'_production_mode': False,
               '_start_ts_override': dt.datetime.utcnow() - dt.timedelta(days=10),
               '_end_ts_override': (dt.datetime.utcnow() - dt.timedelta(days=1)),  # .strftime('%Y-%m-%d %H:%M:%S'),
               '_db_schema': 'BLUADMIN',
               'save_trace_to_file': True}

logger.info('Instantiated create compressor job')

job = JobController(meta, **jobsettings)

entity.exec_local_pipeline()

print ( "Read Table of new  entity" )
df = db.read_table(table_name=entity_type_name, schema=db_schema)
print(df.head())
'''

print ( "Done registering  entity" )


#  Allows you to run and test one function locally.
fn = State_Timer(state_column='running_status', state_name="RUNNING" ,state_metric_name='running_minutes')
df = fn.execute_local_test(db=db, db_schema=db_schema, generate_days=1,to_csv=True)
print(df)


'''
df = pd.DataFrame({'evt_timestamp': [pd.Timestamp('2020-04-10 07:26:14.687196'),
                                                  pd.Timestamp('2020-04-10 07:31:14.687196'),
                                                  pd.Timestamp('2020-04-10 07:46:14.687196'),
                                                  pd.Timestamp('2020-04-05 21:27:04.209610'),
                                                  pd.Timestamp('2020-04-05 21:32:04.209610'),
                                                  pd.Timestamp('2020-04-05 21:37:04.209610'),
                                                  pd.Timestamp('2020-04-05 21:42:09.209610'),
                                                  pd.Timestamp('2020-04-10 07:51:14.687196'),
                                                  pd.Timestamp('2020-04-10 08:00:14.687196')],
                                'drvn_p1': [19.975879, 117.630665, 17.929952, 1.307068,
                                            0.653883, 0.701709, 0.701709, 16.500000, 16.001709],
                                'deviceid': ['73001', '73001', '73001', '73000',
                                             '73000', '73000', '73000', '73001', '73001'],
                                'drvr_rpm': [165, 999, 163, 30,
                                             31, 33, 33, 150, 149],
                                'runningstatus': ["RUNNING", "STOPPED", "RUNNING", "RUNNING",
                                                  "STOPPED", "RUNNING", "RUNNING", "STOPPED", "RUNNING"]
                                },
                               index=[0, 1, 2, 3,
                                      4, 5, 6, 7, 8])

fn.execute_local_test(db=db, db_schema=db_schema, df=df)
'''



'''
meta = db.get_entity_type(entityType)
jobsettings = {}
jobsettings = {'_production_mode': False,
               '_start_ts_override': dt.datetime.utcnow() - dt.timedelta(days=10),
               '_end_ts_override': (dt.datetime.utcnow() - dt.timedelta(days=1)),  # .strftime('%Y-%m-%d %H:%M:%S'),
               '_db_schema': 'BLUADMIN',
               'save_trace_to_file': True}

logger.info('Instantiated create compressor job')

job = JobController(meta, **jobsettings)
job.execute()

entity.exec_local_pipeline()
'''