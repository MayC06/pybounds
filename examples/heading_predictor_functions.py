# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:34:24 2025

Functions to preprocess trajectory variables and predict heading from ground-
speed, thrust, and airspeed plus associated angles run through a neural 
network model (e.g. model_CEM_all-angle-rotate.keras).

@author: mayc06
"""

import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.ndimage import gaussian_filter1d


def augment_with_time_delay_embedding(fly_traj_list: list[pd.DataFrame],**kwargs):
    
    def collect_offset_rows(df, aug_column_names=None, keep_column_names=None, w=1, direction='backward'):
        """ Takes a pandas data frame with n rows, list of columns names, and a window size w.
            Then creates an augmented data frame that collects prior or future rows (in window)
            and stacks them as new columns. The augmented data frame will be size (n - w - 1) as the first/last
            w rows do not have enough data before/after them.

            Inputs
                df: pandas data frame
                aug_column_names: names of the columns to augment
                keep_column_names: names of the columns to keep, but not augment
                w: lookback window size (# of rows)
                direction: get the rows from behind ('backward') or front ('forward')

            Outputs
                df_aug: augmented pandas data frame.
                        new columns are named: old_name_0, old_name_1, ... , old_name_w-1
        """

        df = df.reset_index(drop=True)

        # Default for testing
        if df is None:
            df = np.atleast_2d(np.arange(0, 11, 1, dtype=np.double)).T
            df = np.matlib.repmat(df, 1, 4)
            df = pd.DataFrame(df, columns=['a', 'b', 'c', 'd'])
            aug_column_names = ['a', 'b']
        else:  # use the input  values
            # Default is all columns
            if aug_column_names is None:
                aug_column_names = df.columns

        # Make new column names & dictionary to store data
        new_column_names = {}
        df_aug_dict = {}
        for a in aug_column_names:
            new_column_names[a] = []
            df_aug_dict[a] = []

        for a in aug_column_names:  # each augmented column
            for k in range(w):  # each point in lookback window
                new_column_names[a].append(a + '_' + str(k))

        # Augment data
        n_row = df.shape[0]  # # of rows
        n_row_train = n_row - w + 1  # # of rows in augmented data
        for a in aug_column_names:  # each column to augment
            data = df.loc[:, [a]]  # data to augment
            data = np.asmatrix(data)  # as numpy matrix
            df_aug_dict[a] = np.nan * np.ones((n_row_train, len(new_column_names[a])))  # new augmented data matrix

            # Put augmented data in new column, for each column to augment
            for i in range(len(new_column_names[a])):  # each column to augment
                if direction == 'backward':
                    # Start index, starts at the lookback window size & slides up by 1 for each point in window
                    startI = w - 1 - i

                    # End index, starts at end of the matrix &  & slides up by 1 for each point in window
                    endI = n_row - i  # end index, starts at end of matrix &

                elif direction == 'forward':
                    # Start index, starts at the beginning of matrix & slides up down by 1 for each point in window
                    startI = i

                    # End index, starts at end of the matrix minus the window size
                    # & slides down by 1 for each point in window
                    endI = n_row - w + 1 + i  # end index, starts at end of matrix &

                else:
                    raise Exception("direction must be 'forward' or 'backward'")

                # Put augmented data in new column
                df_aug_dict[a][:, i] = np.squeeze(data[startI:endI, :])

            # Convert data to pandas data frame & set new column names
            df_aug_dict[a] = pd.DataFrame(df_aug_dict[a], columns=new_column_names[a])

        # Combine augmented column data
        df_aug = pd.concat(list(df_aug_dict.values()), axis=1)

        # Add non-augmented data, if specified
        if keep_column_names is not None:
            for c in keep_column_names:
                if direction == 'backward':
                    startI = w - 1
                    endI = n_row
                elif direction == 'forward':
                    startI = 0
                    endI = n_row - w
                else:
                    raise Exception("direction must be 'forward' or 'backward'")

                keep = df.loc[startI:endI, [c]].reset_index(drop=True)
                df_aug = pd.concat([df_aug, keep], axis=1)

        return df_aug
    time_window = kwargs["time_window"]
    input_names = kwargs["input_names"]
    output_names = kwargs["output_names"]
    direction = kwargs["direction"]
    traj_augment_list = []
    for traj in fly_traj_list:
        traj_augment = collect_offset_rows(traj,
                                           aug_column_names=input_names,
                                           keep_column_names=output_names,
                                           w=time_window,
                                           direction=direction)

        traj_augment_list.append(traj_augment)

    traj_augment_all = pd.concat(traj_augment_list, ignore_index=True)

    return np.round(traj_augment_all, 4)


def predict_heading_from_fly_trajectory(df: pd.DataFrame, 
                                        n_input, 
                                        augment_with_time_delay_embedding: callable, 
                                        estimator: callable, 
                                        smooth=False, 
                                        **kwargs):
    """
    Predict heading angles from a trajectory using a trained model.

    Parameters:
    - df (pd.DataFrame): DataFrame containing trajectory data.
    - n_input (int): Number of input features for prediction.
    - augment_with_time_delay_embedding (callable): Function to augment data with time-delay embedding.
    - estimator (callable): Trained model for prediction.
    - smooth (bool): Whether to apply Gaussian smoothing to predicted headings. Default is False.

    Returns:
    - np.ndarray: Array of predicted heading angles.
    """
    augmented_df = augment_with_time_delay_embedding([df], **kwargs)
    augmented_df = augmented_df.iloc[:, 0:n_input]
    print(n_input)
    heading_components = estimator.predict(augmented_df)

    if smooth:
        # Apply Gaussian smoothing
        sigma = 2  # Standard deviation for the Gaussian filter
        heading_components = gaussian_filter1d(heading_components, sigma=sigma, axis=0)

    heading_angle_predicted = np.arctan2(heading_components[:, 1], heading_components[:, 0])

    # Handle time steps removed by augmentation
    num_deleted_steps = len(df["position_x"]) - len(heading_angle_predicted)
    prepend_values = np.full(num_deleted_steps, heading_angle_predicted[0])
    heading_angle_predicted_arr = np.concatenate([prepend_values, heading_angle_predicted])

    return heading_angle_predicted_arr