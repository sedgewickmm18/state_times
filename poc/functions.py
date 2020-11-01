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
        print ( 'state_column= %s  state_metric_name= %s ' %(state_column, state_metric_name) )
        # Input column that has status of the system liked stopped or running  0 or 1
        self.state_column = state_column
        # Output column  metric_name that  time in minutes will be returned for
        self.state_metric_name = state_metric_name
        # Output column   time in minutes for state of metric_name
        #self.state_times = state_time
        self.states = []
        super().__init__()

    def execute(self, df ):
        print("start ----- %s " % self.state_column)
        simulation_data = df
        print(simulation_data.head())

        # List unique values in the df['name'] column
        print('List of Running Status')
        states = simulation_data[self.state_column].unique()
        print(states)

        # Initialize status you need to find running times for
        pd.set_option('display.max_columns', None)
        for state in states:
            simulation_data[state] = 0

        print("Original Simulation Data to test downtime calculations")
        orig_simulation_data = simulation_data
        print(simulation_data)

        print("List of unique equipment")
        asset_list = simulation_data['deviceid'].unique().tolist()
        print(asset_list)

        for asset in asset_list:
            print("Get rows just for device %s --" % asset)
            df_out = simulation_data.loc[simulation_data['deviceid'] == asset]
            print(df_out)
            rows = [list(r) for i, r in df_out.iterrows()]
            first_row = True

            for row in rows:
                if first_row == False:
                    print("Row")
                    print(row)
                    print("-------laststatus_timestamp %s" % laststatus_timestamp)

                    # Check what state row is in and calculate mins_running
                    for item in states:
                        print("Checking if current row  %s is in state %s" % (row[4], item))
                        if row[4] == item:
                            print("Match  %s current time  %s and last status time %s" % (
                            item, row[0], laststatus_timestamp))
                            mins_running = row[0] - laststatus_timestamp
                            print("mins %s" % item)
                            print(mins_running.total_seconds() / 60)
                            # Update original dataframe with calculated minutes running
                            simulation_data.loc[
                                (simulation_data['deviceid'] == asset) & (simulation_data['evt_timestamp'] == row[0]), [
                                    item]] = mins_running.total_seconds() / 60
                else:
                    first_row = False
                print("Last status_timestamp %s " % row[0])
                laststatus_timestamp = row[0]

            for item in states:
                print("\n -- %s Device total mins running in state %s -- \n" % (asset, item))
                print(simulation_data.loc[simulation_data['deviceid'] == asset, item].sum())
                print("\n ---- \n")

        print('simulation_data------')
        print( simulation_data.head() )
        print('orig_simulation_data-----')
        print( orig_simulation_data.head() )
        print('column of minutes being returned-----')
        simulation_data[self.state_metric_name]

        return simulation_data[self.state_metric_name]


    @classmethod
    def build_ui(cls):

        inputs = []
        inputs.append(UISingleItem(name='state_column', datatype=float, description='Name of column you want to measure state time in minutes.'))
        inputs.append(ui.UISingle(name='state_metric_name', datatype=str, description='State name to measure state time in minutes.') )


        # Add output aruguments for each state metric_Name
        outputs = []
        outputs.append(ui.UIFunctionOutSingle(name='state_metric_name', datatype=float, description='Minutes in state'))

        return inputs, outputs
