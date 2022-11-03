import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Download data with kaggel API
def download_data(dataset_name, data_dir):
    # check kaggle installed
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".kaggle")):
        logging.error("Kaggle is not installed. Please install kaggle first.")
        return

    os.system(f"kaggle competitions download -c {dataset_name} -p {data_dir}")
    logging.info(f"Downloaded {dataset_name} from kaggle to {data_dir}")
    os.system(f"unzip {data_dir}/{dataset_name}.zip -d {data_dir}")
    logging.info(f"Unzipped {dataset_name} to {data_dir}")
    os.system(f"rm {data_dir}/{dataset_name}.zip")
    logging.info(f"Removed {dataset_name}.zip")


def basic_preprocess(df, train=True, target="transported"):
    '''
    Basic preprocessing for the dataset
    input: 
        df: pandas dataframe
        train: boolean, if True, then it will drop the target column
    output:
        df: cleaned pandas dataframe
        cat_cols: list of categorical columns
        num_cols: list of numerical columns
        target: target column name
    '''

    df = df.copy()
    #target = 'transported'

    # replacing ' ' and '-' with '_' in column names and converting to lower case
    logging.info('Replacing " " and "-" with "_" in and  column names and converting to lower case')
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    target = target.lower().replace(' ', '_').replace('-', '_')

    # converting CryoSleep to binary
    logging.info('Converting CryoSleep to binary')
    df['cryosleep'] = df['cryosleep'].map({True: 1, False: 0})
    
    # converting VIP to binary
    logging.info('Converting VIP to binary')
    df['vip'] = df['vip'].map({True: 1, False: 0})

    # splitting 'Cabin' from `deck/num/side` to only `deck`
    logging.info('Splitting "cabin" from `deck/num/side` to `deck` `number` and `side`')
    df['cabin_deck'] = df['cabin'].str.split('/').str[0]
    df['cabin_number'] = df['cabin'].str.split('/').str[1].astype('float')
    df['cabin_side'] = df['cabin'].str.split('/').str[2]
    df.drop('cabin', axis=1, inplace=True)

    # categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    logging.info(f"Found {len(cat_cols)} categorical columns")
    logging.info(f"Categorical columns: {cat_cols}")

    # numerical columns
    num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    logging.info(f"Found {len(num_cols)} numerical columns")
    logging.info(f"Numerical columns: {num_cols}")

    # filter object columns and convert to lower case
    logging.info('Converting str to lower case')
    df[cat_cols] = df[cat_cols].apply(lambda x: x.str.lower())

    # only for train data
    if train:
        # converting target to binary
        logging.info('Converting target to binary')
        df[target] = df[target].map({True: 1, False: 0})
        
    # remove target from cat_cols or num_cols
    logging.info('Removing target from cat_cols or num_cols')
    if target in cat_cols:
        cat_cols = [col for col in cat_cols if col != target]
    if target in num_cols:
        num_cols = [col for col in num_cols if col != target]

    # Fill categorical columns with mode
    logging.info('Filling categorical columns with mode')
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Fill numerical columns with mean
    logging.info('Filling numerical columns with mean')
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    cat_cols_to_use = ['homeplanet', 'destination', 'cabin_deck', 'cabin_side']
    df['cryosleep'] = df['cryosleep'].astype('category')
    df['vip'] = df['vip'].astype('category')

    return df, cat_cols_to_use, num_cols, target


if __name__ == "__main__":
    download_data("spaceship-titanic", "./data/")

    