# Create Demo Entity to demonstrate anomaly detection with dimensional filters
# See https://github.com/ibm-watson-iot/functions/blob/development/iotfunctions/entity.py
from iotfunctions import metadata
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func, SmallInteger
from iotfunctions import bif
import logging
from iotfunctions.enginelog import EngineLogging
EngineLogging.configure_console_logging(logging.DEBUG)
logger = logging.getLogger(__name__)


class Equipment (metadata.BaseCustomEntityType):
    '''
    Sample entity type for monitoring Equipment.
    '''

    def __init__(self,
                 name,
                 db,
                 db_schema=None,
                 # functions=functions,
                 # columns=columns,
                 description=None,
                 generate_days=0,
                 drop_existing=True,
                 ):

        # constants
        constants = []

        physical_name = name.lower()

        # granularities
        granularities = []

        # columns
        columns = []

        columns.append(Column('runningstatus', String(50) ))
        columns.append(Column('drvr_rpm', Float() ))
        columns.append(Column('drvn_p1', Float() ))

        # dimension columns
        dimension_columns = []

        # functions
        functions = []

        # simulation settings
        # uncomment this if you want to create entities automatically
        # then comment it out
        # then delete any unwanted dimensions using SQL
        #   DELETE FROM BLUADMIN.EQUIPMENT WHERE DEVICEID=73000;
        sim = {
            'freq': '5min',
            'auto_entity_count' : 1,
            'data_item_mean': {'drvr_rpm': 1,
                               'drvn_p1': 50,
                               'runningstatus': 1
                               },
            'data_item_domain': {
            },
            'drop_existing': True
        }

        generator = bif.EntityDataGenerator(ids=None, parameters=sim)
        functions.append(generator)

        # data type for operator cannot be inferred automatically
        # state it explicitly

        output_items_extended_metadata = {}

        super().__init__(name=name,
                         db = db,
                         constants = constants,
                         granularities = granularities,
                         columns=columns,
                         functions = functions,
                         dimension_columns = dimension_columns,
                         output_items_extended_metadata = output_items_extended_metadata,
                         generate_days = generate_days,
                         drop_existing = drop_existing,
                         description = description,
                         db_schema = db_schema)
