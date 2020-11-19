import logging
import pandas as pd
from iotfunctions import ui

from iotfunctions.base import BaseTransformer
#from .ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti)
from iotfunctions.ui import UIFunctionOutSingle
from iotfunctions.enginelog import EngineLogging
EngineLogging.configure_console_logging(logging.DEBUG)

logger = logging.getLogger(__name__)

# Specify the URL to your package here.
# This URL must be accessible via pip install

PACKAGE_URL = 'git+https://github.com/carlosibm/state_times'


class State_Timer(BaseTransformer):
    '''
    For a selected metric calculates the amount of time in minutes it has been in that  state since the last change in state.
    '''

    def __init__(self, state_column, state_metric_name):
        logger.debug( 'state_column= %s  state_metric_name= %s ' %(state_column, state_metric_name) )
        # Input column that has status of the system liked stopped or running  0 or 1
        self.state_column = state_column
        # Output column  metric_name that  time in minutes will be returned for
        self.state_metric_name = state_metric_name
        # Output column   time in minutes for state of metric_name
        #self.state_times = state_time
        self.states = []
        super().__init__()

    def execute(self, df ):
        logger.debug("Original Simulation Data to test downtime calculations")
        logger.debug("start df ----- %s " % df)
        logger.debug("start df ----- %s " % df.columns)
        logger.debug("start state_column  ----- %s " %self.state_column)
        logger.debug("start print state_metric_name %s " %self.state_metric_name)

        # List unique values in the df['name'] column
        logger.debug('List of Running Status')
        states = df[self.state_column].unique()
        logger.debug(states)

        logger.debug("Original Simulation Data looking at rows")
        for index, row in df.iterrows():
            logger.debug("original rows")
            logger.debug(row)

        # Initialize status you need to find running times for
        pd.set_option('display.max_columns', None)
        for state in states:
            df[self.state_metric_name] = 0
            df[state] = 0

        logger.debug("Debugging")
        entity_index_name = df.index.names[0]
        time_index_name = df.index.names[1]
        df.reset_index(inplace=True)
        logger.debug('Here are entity_index_name, time_index_name, df.columns ', entity_index_name, time_index_name, df.columns)
        asset_list = df[entity_index_name].unique().tolist()
        logger.debug("List of unique equipment")
        logger.debug(asset_list)

        logger.debug("Analyze Index")
        for asset in asset_list:
            logger.debug("Get rows just for single asset %s --" % asset)
            df_asset = df.loc[df[entity_index_name] == asset]
            logger.debug("Rows just for %s" %asset )
            logger.debug(df_asset)

            # rows = [list(r) for i, r in df_asset.iterrows()]
            first_row = True
            # self.state_column_idx = list(df_asset.columns).index(self.state_column)

            for index, row in df_asset.iterrows():
                if first_row == False:
                    logger.debug("iterate rows")
                    logger.debug(row['evt_timestamp'], row[df[entity_index_name]], row[self.state_column])

                    # Calculate mins running
                    mins_running = row['evt_timestamp'] - laststatus_timestamp
                    laststatus_timestamp = row['evt_timestamp']
                    logger.debug("New status_timestamp %s " % laststatus_timestamp)
                    logger.debug("mins_running %s " % mins_running)
                    # Update original dataframe with calculated minutes running
                    df.loc[
                        (df[entity_index_name] == asset) & (
                                    df['evt_timestamp'] == row['evt_timestamp']), [
                            row[self.state_column]]] = mins_running.total_seconds() / 60

                    # df.loc[(df['deviceid'] == asset) & (df['evt_timestamp'] == row['evt_timestamp'], df[self.state_metric_name]  = mins_running.total_seconds() / 60
                else:
                    logger.debug("First Row")
                    logger.debug(row['evt_timestamp'], row[df[entity_index_name]], row[self.state_column])
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
            logger.debug("Rows just for %s" %asset )
            logger.debug(df_asset)

        # Reset DF the index back to what it was
        df.set_index([entity_index_name, time_index_name], inplace=True)

        logger.debug('Column we are returning with state_metric_name and minutes |||  ')
        logger.debug(states)
        logger.debug(df[states])

        logger.debug('Final entire DF we are returning with state_metric_name and minutes |||  ')
        logger.debug(df)
        return df

    @classmethod
    def build_ui(cls):

        inputs = []
        inputs.append(ui.UISingleItem(name='state_column', datatype=str, description='Name of column (status)  you want to measure state time in minutes.'))

        inputs.append(ui.UISingle(name='state_metric_name', datatype=str,  description='Enter name of the state to measure time of'))

        '''
        inputs.append(ui.UIMultiItem(name='state_metric_names',
                                    datatype=float,
                                    required=True,
                                    description='State name (running) to measure state time in minutes.',
                                    output_item='predictions',
                                    is_output_datatype_derived=True
                                  ))
        aggregate_names = list(cls.get_available_methods().keys())


        inputs.append(UIMultiItem(name='targets',
                                  datatype=float,
                                  required=True,
                                  output_item='predictions',
                                  is_output_datatype_derived=True
                                  ))
        '''

        # Add output aruguments for each state metric_Name
        outputs = []
        outputs.append(ui.UISingle(name='state_metric_name', datatype=float, description='Minutes in state'))
        return (inputs, outputs)

