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

def _get_libsvm(target, feat, set_name):

    M = feat.as_matrix()
    with open(set_name+'.txt', 'w') as f:
        # Get matrix shape and set range settings for iterating over rows
        _, row_len = M.shape
        row_range = range(row_len)

        for row in M:
            #write label/rank value
            for i in row_range:
                #write index:value \s for every nonzero value
            
            # end with eol





def parse_data(filepath, train_name='train', test_name='test'):
    df = pd.read_csv('DMT_train.csv', header = 0, index_col=0)
    print(df.head())

    #########################################
    # Get dummy variables
    #########################################

    cat_feats = ['visitor_location_country_id', 'prop_country_id', 'srch_destination_id']

    for feat in cat_feats:
        pd.get_dummies(df[feat], prefix=feat)

    #########################################
    # Remove unnecessary variables
    #########################################

    df = df.drop(columns = ['srch_id', 'prop_id', 'date_time'])


    #########################################
    # Sort data on position for group input format file
    #########################################

    df.sort_values('position', inplace=True)

    

parse_data('DMT_train.csv')