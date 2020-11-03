import logging
import pandas as pd
from iotfunctions import ui

from iotfunctions.base import BaseTransformer
#from .ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti)
from iotfunctions.ui import UIFunctionOutSingle
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
        logger.debug("start state_column ----- %s " % self.state_column)
        logger.debug("start state_metric_name ----- %s " % self.state_metric_name)

        simulation_data = df
        logger.debug(simulation_data.head())

        # List unique values in the df['name'] column
        logger.debug('List of Running Status')
        states = simulation_data[self.state_column].unique()
        logger.debug(states)

        # Initialize metric name state with 0 minutes
        pd.set_option('display.max_columns', None)
        for state in states:
            simulation_data[state] = 0

        logger.debug("Original Simulation Data to test downtime calculations")
        orig_simulation_data = simulation_data
        simulation_data[self.state_metric_name] = 0
        logger.debug(simulation_data)

        logger.debug("List of unique equipment")
        entity_index_name = simulation_data.index.names[0]
        time_index_name = simulation_data.index.names[1]
        simulation_data.reset_index(inplace=True)
        logger.debug('Here', entity_index_name, time_index_name, simulation_data.columns)
        asset_list = simulation_data[entity_index_name].unique().tolist()
        logger.debug(asset_list)

        for asset in asset_list:
            logger.debug("Get rows just for device %s --" % asset)
            df_out = simulation_data.loc[simulation_data[entity_index_name] == asset]
            logger.debug(df_out)
            rows = [list(r) for i, r in df_out.iterrows()]
            first_row = True
            state_column_idx = list(df_out.columns).index(self.state_column)

            for row in rows:
                if first_row == False:
                    logger.debug("Row")
                    logger.debug(row)
                    logger.debug("-------laststatus_timestamp %s" % laststatus_timestamp)

                    # Check what state row is in and calculate mins_running
                    for item in states:
                        logger.debug("Checking if current row  %s is in state %s" % (state_column_idx, item))
                        if row[4] == item:
                            logger.debug("Match  %s current time  %s and last status time %s" % (
                                item, row[1], laststatus_timestamp))
                            mins_running = row[1] - laststatus_timestamp
                            logger.debug("mins %s" % item)
                            logger.debug(mins_running.total_seconds() / 60)
                            # Update original dataframe with calculated minutes running
                            simulation_data.loc[
                                (simulation_data[entity_index_name] == asset) & (
                                            simulation_data[time_index_name] == row[1]), [
                                    item]] = mins_running.total_seconds() / 60
                else:
                    first_row = False
                logger.debug("Last status_timestamp %s " %row[1])
                laststatus_timestamp = row[1]

            for item in states:
                logger.debug("\n -- %s Device total mins running in state %s -- \n" % (asset, item))
                logger.debug(simulation_data.loc[simulation_data[entity_index_name] == asset, item].sum())
                logger.debug("\n ---- \n")

        logger.debug('simulation_data------')
        logger.debug( simulation_data.head() )
        logger.debug('orig_simulation_data-----')
        logger.debug( orig_simulation_data.head() )
        logger.debug('column of minutes being returned-----')
        simulation_data.set_index([entity_index_name, time_index_name], inplace=True)
        return simulation_data[self.state_metric_name]


    @classmethod
    def build_ui(cls):

        inputs = []
        inputs.append(ui.UISingleItem(name='state_column', datatype=str, description='Name of column (status)  you want to measure state time in minutes.'))
        inputs.append(ui.UISingle(name='state_metric_name', datatype=str, description='State name (running) to measure state time in minutes.') )


        # Add output aruguments for each state metric_Name
        outputs = []
        outputs.append(ui.UIFunctionOutSingle(name='state_metric_name', datatype=float, description='Minutes in state'))

        return inputs, outputs
