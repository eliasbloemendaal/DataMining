import pandas as pd
import time


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

def _get_libsvm(df, set_name, train = True):
    """
        Writes a SORTED df to libsvm format
    """
    t1 = time.time()
    with open('datasets/'+set_name+'.txt', 'w') as f:
        # Get matrix shape and set range settings for iterating over rows

        for _, row in df.iterrows():
            #write label/rank value
            row_elements = []
            if train:
                row_elements.append(str(row.position))
                row.drop('position', inplace = True)
            row = row.tolist()
            #get all nonzero values and their indices
            for index, val in enumerate(row):
                if val != 0:
                    row_elements.append('{}:{}'.format(index, val))
            
            #Write libsvm row format to file
            f.write(' '.join(row_elements))
            f.write('\n')
            # end with eol
        print(df.shape)
        print(time.time()-t1)

def _get_group_input(df, set_name):
    """
        Creates a file describing the number of elements per group for a SORTED df
    """
    group_counts = df.groupby('srch_id')['prop_id'].count()
    filename = 'datasets/{}.txt.group'.format(set_name)

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
    filename = 'datasets/{}.txt.weight'.format(set_name)
    print('set {}, sum weights: {}'.format(set_name, sum(weights.values)))

    weights.to_csv(filename, index=None, header=None)

def _prep_train(df):
    # Bound outliers

    print(df.shape)
    # Downsample negative values
    nr_srch_ids = len(df.srch_id.tolist())
    #srch_id_sum = df.groupby('srch_id')['booking_bool', 'click_bool'].sum().reset_index()
    #print('hier')
    #print(srch_id_sum)

    non_booked_clicked = df[(df.booking_bool == 0) & (df.click_bool == 0)]
    nr_negative = non_booked_clicked.shape[0]
    #print('total_rows {}'.format(srch_id_sum.shape[0]))
    #print('#negative:{}'.format(nr_negative))
    #print('#positive:{}'.format(srch_id_sum.shape[0]-nr_negative))
    ratio_pos = (nr_srch_ids-nr_negative)/nr_srch_ids
    desired_ratio = 0.15
    print('RATIO positive values: {}'.format(ratio_pos))
    keep_prob = ratio_pos / desired_ratio

    reduced_non_booked_clicked = non_booked_clicked.sample(frac=keep_prob, random_state = 42)
    del non_booked_clicked

    df = pd.concat([df[df.srch_id.isin(reduced_non_booked_clicked.srch_id.tolist())],df[(df.booking_bool == 1) | (df.click_bool == 1)]])

    return df

def _prep_files(df, name, train=True):
    #########################################
    # Sort data on position for group input format file
    #########################################

    df = df.sort_values('srch_id', kind = 'heapsort')


    #########################################
    # Get group information
    #########################################
    _get_group_input(df, name)
    _get_instance_weight(df, name)

    #########################################
    # Remove unnecessary variables
    #########################################

    df = df.drop(columns = ['srch_id', 'prop_id', 'date_time', 'min_date', 'max_date', 'day'])
    if train:
        df = df.drop(columns = ['booking_bool', 'click_bool', 'gross_bookings_usd'])

    #########################################
    # Get instance information in libsvm format
    #########################################
    _get_libsvm(df, name)
    return df

def parse_data(filepath, train_name='train', test_name='test'):
    df = pd.read_csv(filepath, header = 0, index_col=0)
    #print(df.head())

    #########################################
    # create subsets
    #########################################

    unique_ids = pd.Series(df.srch_id.unique())

    # final train set (all data)
    #final_train = _prep_train(df)
    # result = _prep_files(final_train, 'final_train')
    # result.to_csv('datasets/final_train.csv', index = None )

    # del final_train
    #del result
    
    # cv train set (80 %)
    cv_train_ids = unique_ids.sample(frac=0.8, random_state = 42)
    # cv_train = df[df.srch_id.isin(cv_train_ids.tolist())]
    # cv_train = _prep_train(cv_train)
    # result = _prep_files(cv_train, 'cv_train')
    # result.to_csv('datasets/cv_train.csv', index = None )

    # del cv_train
    # del unique_ids
    #del result


    # test set (20 %)
    # test_set = df[~df.srch_id.isin(cv_train_ids.tolist())]
    # result = _prep_files(test_set, 'test_set')
    # result.to_csv('datasets/test_set.csv', index = None )

    # del test_set
    # del result
    

    # GA train set (64 %)
    GA_train_ids = cv_train_ids.sample(frac=0.8, random_state = 42)
    # GA_train = df[df.srch_id.isin(GA_train_ids.tolist())]
    # GA_train = _prep_train(GA_train)
    # result = _prep_files(GA_train, 'GA_train')
    # result.to_csv('datasets/GA_train.csv', index = None )

    # del GA_train
    # del result

    # input('hier')


    # GA validation set ( 16%)
    GA_valid = df[~df.srch_id.isin(GA_train_ids)]
    del df
    result = _prep_files(GA_valid, 'GA_valid')
    result.to_csv('datasets/GA_valid.csv', index = None )

    input('hier')
    ###########################################
    # Prep files
    ###########################################
    #(final_train, 'final_train'), (cv_train, 'cv_train'), (test_set, 'test_set'),

    for dataset in [ (GA_train, 'GA_train'), (GA_valid, 'GA_valid')]:
        result = _prep_files(dataset[0], dataset[1])
        result.to_csv('datasets/'+dataset[1]+'.csv', index = None )

# df1 = pd.read_csv('train_neg_downsampled.csv', header=0, index_col=0)
# df2 = pd.read_csv('train_pos.csv', header=0, index_col=0)

# pd.concat([df1,df2]).to_csv('downsampled_train.csv')

# input('hier')

# df = pd.read_csv('DMT_train_full.csv', header = 0, index_col=0)
# print(df.shape)


# ratio_pos = 0.04474858254172207
# keep_prob = ratio_pos/0.010

# non_booked_clicked = df[(df.booking_bool == 0) & (df.click_bool == 0)]
# print(non_booked_clicked.shape)
# non_booked_clicked.sample(n=1341000, random_state = 42).to_csv('train_neg_downsampled.csv')


    
#Use for train sets
#parse_data('downsampled_train.csv')
#use for valid/train sets
parse_data('DMT_train_full.csv')