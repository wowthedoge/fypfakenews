import pandas as pd

from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, bigquery_storage
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ID = "mega-mind-379101"
DATASET_ID = "customer_persona"
ARTICLE_SIMILARITY_TABLE = "article_similarity_table"
LOOKBACK_PERIOD_MONTH = 3

# Initialize the BigQuery client
client = bigquery.Client(project=PROJECT_ID)

# Initialize the BigQuery Storage API client
bq_storage_client = bigquery_storage.BigQueryReadClient()

sql = f"""
SELECT 
    articleID, 
    language, 
    title, 
    published_date 
FROM 
    `mega-mind-379101.customer_persona.article_meta` 
WHERE 
    DATE_DIFF(CURRENT_DATE(), DATE(published_date), MONTH) <= {LOOKBACK_PERIOD_MONTH}
"""

###################################################################
# Load Data from BigQuery                                         #
###################################################################

print("Downloading data from BigQuery...")
df = client.query(sql).to_dataframe(bqstorage_client=bq_storage_client)
print(df.shape)

###################################################################
# NULL value Imputation & Data Cleaning                           #
###################################################################

print("Imputing null values and further preprocessing...")
null_imputation_dict = { 
    'articleID': 'None',
    'language': 'None',
    'title': 'None'
    }

df = df.fillna(value=null_imputation_dict)

###################################################################
# Encode Article to Embedding Vectors (Feature Extraction)        #
###################################################################

print("Encoding articles into vectors...")
# Load a pre-trained multilingual model
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Ref: https://www.sbert.net/docs/package_reference/SentenceTransformer.html
embeddings = model.encode(df['title'], batch_size=32, show_progress_bar=True)

print("Converting article embeddings into a DataFrame object...")
# Convert embeddings to a dataframe
embeddings_df = pd.DataFrame(embeddings)

# Join the original dataframe with the embeddings dataframe
df_article_embeddings = pd.concat([df[['articleID']], embeddings_df], axis=1)

# Get index-articleID mapping to be used later
idx_articleID_mapping = df_article_embeddings['articleID'].to_dict()

# Choose columns needed to build similarity index.
df_article_embeddings.drop('articleID', axis=1, inplace=True)
print(df_article_embeddings.shape)

###################################################################
# Build Article Similarity Map using ANNOY                        #
###################################################################

print("Building similarity map...")
def build_annoy_index(df_encoded, num_trees=10):
    """
    Builds an Annoy index from the preprocessed DataFrame.

    Args:
    df_encoded (DataFrame): A preprocessed DataFrame with categorical variables encoded and numerical variables normalized.
    num_trees (int): The number of trees to build for the Annoy index. A higher value increases accuracy but decreases performance.

    Returns:
    AnnoyIndex: An Annoy index object that can be used for querying approximate nearest neighbors.
    """
    f = df_encoded.shape[1]  # Get the number of features from the preprocessed DataFrame
    index = AnnoyIndex(f, 'angular')  # Create an Annoy index with the specified number of features and distance metric
    
    for i, row in df_encoded.iterrows():  # Add each row of the preprocessed DataFrame to the Annoy index
        index.add_item(i, row.values)
        
    index.build(num_trees)  # Build the Annoy index using the specified number of trees
    
    return index

def get_top_100_similar_articles(article_id, annoy_index):
    """
    Returns the top 100 similar articles and their similarity scores for a given articleId.

    Args:
    article_id (int): The ID of the article for which to find the top 100 similar articles.
    annoy_index (AnnoyIndex): The Annoy index built from the preprocessed DataFrame.

    Returns:
    tuple: A tuple containing the articleID and a list of tuples, where each tuple contains a similar article ID and its similarity score.
    """
    # Get the top 101 nearest neighbors (including the article itself) and their distances
    neighbors, distances = annoy_index.get_nns_by_item(article_id, 101, include_distances=True)

    # Convert distances to similarity scores (1 - distance) and exclude the first neighbor (the article itself)
    similarity_scores = [round(1 - distance, 4) for distance in distances[1:]]
    similar_articles = neighbors[1:]

    # Return the articleId and a list of tuples containing similar articleIds and their similarity scores
    return article_id, list(zip(similar_articles, similarity_scores))

def find_top_100_similar_articles_for_all(df_encoded, num_trees=100, num_workers=None):
    """
    Finds the top 100 similar articles and their similarity scores for each article in the DataFrame using the Annoy index.

    Args:
    df_encoded (DataFrame): A preprocessed DataFrame with categorical variables encoded and numerical variables normalized.
    num_trees (int, optional): The number of trees used to build the Annoy Index.
    num_workers (int, optional): The number of worker threads to use for parallel processing. Defaults to None (use the default number of threads).

    Returns:
    dict: A dictionary where keys are articleIDs and values are lists of tuples containing the top 100 similar articleIDs and their similarity scores.
    """
    annoy_index = build_annoy_index(df_encoded, num_trees=num_trees)  # Build the Annoy index from the preprocessed DataFrame
    
    top_100_similar_articles = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(get_top_100_similar_articles, article_id, annoy_index) for article_id in df_encoded.index]
        
        # for future in tqdm(as_completed(futures), total=len(futures), desc="Finding top 100 similar users"):
        for future in as_completed(futures):
            article_id, similar_articles_and_scores = future.result()
            top_100_similar_articles[article_id] = similar_articles_and_scores
    
    return top_100_similar_articles

print("Finding top 100 similar articles for all articles...")
top_100_similar_articles = find_top_100_similar_articles_for_all(df_article_embeddings, num_trees=200)

###################################################################
# Mapping out articleID                                           #
###################################################################

def map_similarity_scores_to_articleID(idx_articleID_mapping, top_100_similar_articles):
    """
    Maps the similarity scores to the original DataFrame using the index and adds the firebaseID column.

    Args:
    idx_articleID_mapping: The index-articleID mapping recorded from the previous step.
    top_100_similar_articles (dict): A dictionary where keys are articleIDs and values are lists of tuples containing the top 100 similar articleIDs and their similarity scores.

    Returns:
    dict: A dictionary where keys are firebaseIDs and values are lists of tuples containing the top 100 similar firebaseIDs and their similarity scores.
    """
    similar_articles_articleID = {}
    
    for article_id, similar_articles_and_scores in top_100_similar_articles.items():
        articleID = idx_articleID_mapping[article_id]
        similar_articles_articleID[articleID] = [(idx_articleID_mapping[similar_article_id], score) for similar_article_id, score in similar_articles_and_scores]
    
    return similar_articles_articleID

print("Mapping out articleID for all articles...")
similar_articles_articleID = map_similarity_scores_to_articleID(idx_articleID_mapping, top_100_similar_articles)

###################################################################
# Transform and upload final table to BigQuery                    #
###################################################################

print("Transforming data to document format...")
# Function to check if a table exists
def table_exists(client, dataset_id, table_id):
    table_ref = client.dataset(dataset_id).table(table_id)
    try:
        client.get_table(table_ref)
        return True
    except NotFound:
        return False

# Create a dataset reference
dataset_ref = client.dataset(DATASET_ID, project=PROJECT_ID)

# Create the table schema
article_similarity_table_schema = [
    bigquery.SchemaField("articleID", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("article_similarity_scores", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("articleID", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("similarity_score", "FLOAT64", mode="REQUIRED"),
    ]),
]

# Create the table only if it doesn't exist
if not table_exists(client, DATASET_ID, ARTICLE_SIMILARITY_TABLE):
    table = bigquery.Table(dataset_ref.table(ARTICLE_SIMILARITY_TABLE), schema=article_similarity_table_schema)
    table = client.create_table(table)
    print("Table created.")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

transformed_data = []
for key, value_list in tqdm(similar_articles_articleID.items(), desc="Transforming data to document format"):
    transformed_data.append({
        'articleID': key,
        'article_similarity_scores': [
            {'articleID': user_id, 'similarity_score': similarity_score}
            for user_id, similarity_score in value_list
        ],
    })

print("Uploading data to BigQuery...")
# Insert the transformed data into the BigQuery table using the load job config
table_ref = dataset_ref.table(ARTICLE_SIMILARITY_TABLE)

# Load transformed data to BigQuery in chunks
for i, data_chunk in tqdm(enumerate(chunks(transformed_data, 10000)), desc="Uploading chunks to BigQuery"):
    # Create a load job config with WRITE_TRUNCATE disposition for first chunk
    # and WRITE_APPEND disposition for subsequent chunks
    write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE if i == 0 else bigquery.WriteDisposition.WRITE_APPEND
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema=article_similarity_table_schema,
    )
    load_job = client.load_table_from_json(data_chunk, table_ref, job_config=job_config)
    load_job.result()  # Wait for the job to complete

print("Pipeline completed.")