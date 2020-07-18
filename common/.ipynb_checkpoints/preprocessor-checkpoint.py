'''
contains re-indexing logic
and conversion to spark df (optional)
'''
# read in the dataset
import pandas as pd

def create_index_mapping(L):
    '''
    return reindexed dict on user and items
    encoded indices starts from 1
    input: 
    * L: list of str
    outputs:
    * ind_2_item,item_2_ind: tuple of dictionary
    '''
    L = set(L)
    ind_2_item = {}
    
    for i,v in enumerate(L):
        #index start from 1
        ind_2_item[i+1] = v
    #invert the map
    item_2_ind = {v: k for k, v in ind_2_item.items()}
    return ind_2_item,item_2_ind
    
def reindexer(ratings_df,user_col,item_col,rating_col):
    '''
    inputs:
    * ratings_df: pandas df containing ratings/affinity for user-item pairs
    * user_col: actual col name for users
    * item_col: actual col name for items
    * rating_col: actual col name for ratings
    output:
    * ratings_df: reindexed user and item column, pandas df
    '''
    users_list = ratings_df[user_col].tolist()
    item_list = ratings_df[item_col].tolist()
    
    ind_2_user,user_2_ind = create_index_mapping(users_list)
    ind_2_item,item_2_ind = create_index_mapping(item_list)
    
    #rename ratings df
    ratings_df = ratings_df.rename(columns={user_col:'user_col',
                                            item_col:'item_col',
                                            rating_col:'rating_col'})

    #encode df using the 2 mappings
    ratings_df['encoded_users'] = ratings_df['user_col'].apply(lambda x:user_2_ind[x])
    ratings_df['encoded_items'] = ratings_df['item_col'].apply(lambda x:item_2_ind[x])
    
    return ratings_df[['encoded_users','encoded_items','rating_col']]