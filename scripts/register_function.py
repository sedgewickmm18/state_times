import json
import logging
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func

from poc.functions import State_Timer

from iotfunctions.db import Database

from iotfunctions.enginelog import EngineLogging
EngineLogging.configure_console_logging(logging.DEBUG)

import datetime as dt

import sys
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

with open('./credentials_Monitor-Demo2.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())

#with open('../Monitor-Demo-Credentials.json', encoding='utf-8') as F:
#    credentials = json.loads(F.read())

db = Database(credentials = credentials)
db_schema = None #  set if you are not using the default

#db.unregister_functions(["State_Timer"])
logger.debug("Function  unregistered")
# exit()ßß
db.register_functions([State_Timer])
# exit()
#print("Function registered ")
