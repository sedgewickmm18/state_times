import json
import logging
import datetime as dt
from iotfunctions.db import Database
from iotfunctions.enginelog import EngineLogging
from poc.functions import State_TimerV2
import pandas as pd
from scripts.test_entities import Equipment
from iotfunctions.pipeline import JobController

logger = logging.getLogger(__name__)

with open('credentials_Monitor-Demo2.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())
db_schema = 'bluadmin'
db = Database(credentials=credentials)
entity_type_name = 'Container6'
entityType = entity_type_name
# Use create_entities_usingcsv.py to create Entity Type.

#db.drop_table(entity_type_name, schema = db_schema)
entity = Equipment(name = entity_type_name,
                db = db,
                db_schema = db_schema,
                description = "Smart Connect Operations Control Center",
                )

#entity.register(raise_error=False)

meta = db.get_entity_type(entityType)
jobsettings = {'_production_mode': False,
               '_start_ts_override': dt.datetime.utcnow() - dt.timedelta(days=10),
               '_end_ts_override': (dt.datetime.utcnow() - dt.timedelta(days=1)),  # .strftime('%Y-%m-%d %H:%M:%S'),
               '_db_schema': 'BLUADMIN',
               'save_trace_to_file': True}

logger.info('Instantiated create compressor job')

job = JobController(meta, **jobsettings)

entity.exec_local_pipeline()