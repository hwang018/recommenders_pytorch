'''
loading local csv file
contains re-indexing logic
and conversion to spark df (optional)
'''
# read in the dataset
import pandas as pd

def create_index_mapping(L):
    '''
    given a list of items,
    return 2 mappings, from index to item 
    and from item to index
    '''
    L = set(L)
    ind_2_item = {}
    
    for i,v in enumerate(L):
        ind_2_item[i] = v
        
    item_2_ind = {v: k for k, v in ind_2_item.items()}

    return ind_2_item,item_2_ind
    
def load_data(config):
    '''
    load and prepare into pandas df, encode items and users into indexed versions
    consider saving mapping somewhere for future remapping
    first step loading csv data, before being used in pytorch tensors
    '''
    ratings = pd.read_csv(config['ratings'],delimiter=";",encoding= 'unicode_escape')
    users_list = ratings['User-ID'].tolist()
    item_list = ratings['ISBN'].tolist()
    
    ind_2_user,user_2_ind = create_index_mapping(users_list)
    ind_2_item,item_2_ind = create_index_mapping(item_list)
    
    #encode df using the 2 mappings
    ratings['encoded_users'] = ratings['User-ID'].apply(lambda x:user_2_ind[x])
    ratings['encoded_items'] = ratings['ISBN'].apply(lambda x:item_2_ind[x])
    
    return ratings[['encoded_users','encoded_items','Book-Rating']]