import pandas as pd


""" df = pd.read_csv('DMT_train.csv', header = 0, index_col=0)
print(df.head()) """

"""
TODO:
    OHE for categorical features (except srch_id & prop_id? -> drop these + date_time)
        [visitor_location_country_id, prop_country_id, srch_destination_id]
    Also drop gross_booking_usd since it cannot be used for test set

    Sort data on position

    split data into train and test set

    create libsvm format
    create group input format
    create instance weight file acc to Canvas



"""

def _get_libsvm(df, set_name):
    """
        Writes a SORTED df to libsvm format
    """

    with open(set_name+'.txt', 'w') as f:
        # Get matrix shape and set range settings for iterating over rows

        for _, row in df.iterrows():
            #write label/rank value
            row_elements = []
            row_elements.append(str(row.position))
            row.drop('position')
            row = row.tolist()
            #get all nonzero values and their indices
            for index, val in enumerate(row):
                if val != 0:
                    row_elements.append('{}:{}'.format(index, val))
            
            #Write libsvm row format to file
            f.write(' '.join(row_elements))
            f.write('\n')
            # end with eol

def _get_group_input(df, set_name):
    """
        Creates a file describing the number of elements per group for a SORTED df
    """
    group_counts = df.groupby('srch_id')['prop_id'].count()
    filename = '{}.txt.group'.format(set_name)

    group_counts.to_csv(filename, header=None, index=None)

def _get_weight(row):
    #print(row)
    if row.booking_bool == 1: 
        return 5 
    elif row.click_bool == 1:
        return 1
    else:
        return 0

def _get_instance_weight(df, set_name):
    """
        Creates a file with weights for each instance in dataset
        - 5 if prop is booked
        - 1 if prop is clicked
        - 0 else
    """

    weights = df.apply(_get_weight, axis=1)
    filename = '{}.txt.weight'.format(set_name)

    weights.to_csv(filename, index=None, header=None)


def parse_data(filepath, train_name='train', test_name='test'):
    df = pd.read_csv('DMT_train.csv', header = 0, index_col=0)
    #print(df.head())

    #########################################
    # Get dummy variables
    #########################################

    cat_feats = ['visitor_location_country_id', 'prop_country_id', 'srch_destination_id']
    print(len(df.columns))

    # for feat in cat_feats:
    #     df = pd.concat([df, pd.get_dummies(df[feat], prefix=feat)], axis=1)

    print(len(df.columns))

    


    #########################################
    # Sort data on position for group input format file
    #########################################

    df.sort_values('srch_id', inplace=True)


    #########################################
    # Get group information
    #########################################
    _get_group_input(df, 'train')
    _get_instance_weight(df, 'train')

    #########################################
    # Remove unnecessary variables
    #########################################
    print(len(df.srch_id.unique()))

    df = df.drop(columns = ['srch_id', 'prop_id', 'date_time', 'min_date', 'max_date', 'day'])

    #########################################
    # Get instance information in libsvm format
    #########################################
    _get_libsvm(df, 'train')



    

parse_data('DMT_train.csv')