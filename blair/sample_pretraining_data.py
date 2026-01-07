import random
from typing import List
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset


NUM_WORKERS = 1
VALID_TIMESTAMP = 1628643414042
DOWNSAMPLING_FACTOR = 1 # Testing
MIN_TEXT_LENGTH = 30
all_cleaned_item_metadata = {}


def load_all_categories():
    # category_filepath = hf_hub_download(
    #     repo_id='McAuley-Lab/Amazon-Reviews-2023',
    #     filename='all_categories.txt',
    #     repo_type='dataset'
    # )
    # with open(category_filepath, 'r') as file:
    #     all_categories = [_.strip() for _ in file.readlines()]
    # return all_categories
    return ['Appliances']


def concat_item_metadata(dp):
    # print(type(dp))
    # print(dp.keys())
    # print(dp['title'])
    # print(dp['description'])
    # print(dp['features'])
    # print(dp.keys())
    # exit()
    meta_segments: List[str] = []
    if dp.get("title"):
        meta_segments.append(dp["title"])
    if dp.get("features"):
        meta_segments.extend(dp["features"])
    if dp.get("description"):
        meta_segments.extend(dp["description"])
    dp["cleaned_metadata"] = (
        " ".join(meta_segments).replace("\t", " ").replace("\n", " ").replace("\r", "").strip()
    )
    return dp


def filter_reviews(dp, metadata_store):
    # Downsampling: Set factor to 1 temporarily to debug
    if random.randint(1, DOWNSAMPLING_FACTOR) > 1:
        return False
    
    cutoff_date = pd.to_datetime(VALID_TIMESTAMP, unit='ms')
    
    current_ts = pd.to_datetime(dp["timestamp"], unit='ms') if isinstance(dp["timestamp"], int) else pd.to_datetime(dp["timestamp"])
    
    if current_ts >= cutoff_date:
        return False
        
    asin = dp.get('parent_asin')
    # Use the passed-in metadata_store instead of the global variable
    if asin not in metadata_store:
        return False
        
    if len(dp.get('cleaned_review', '')) <= MIN_TEXT_LENGTH:
        return False
        
    return True


def concat_review(dp):
    review_segments: List[str] = []
    if dp.get("title"):
        review_segments.append(dp["title"])
    if dp.get("text"):
        review_segments.append(dp["text"])
    dp["cleaned_review"] = (
        " ".join(review_segments).replace("\t", " ").replace("\n", " ").replace("\r", "").strip()
    )
    return dp


if __name__ == '__main__':
    all_categories = load_all_categories()

    # Load item metadata
    for category in all_categories:
        # meta_dataset = load_dataset(
        #     'McAuley-Lab/Amazon-Reviews-2023',
        #     f'raw_meta_{category}',
        #     split='full'
        #     # trust_remote_code=True
        # )

        data_url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"
        df = pd.read_json(data_url, lines=True)
        meta_dataset = Dataset.from_pandas(df)
        concat_meta_dataset = meta_dataset.map(
            concat_item_metadata,
            num_proc=NUM_WORKERS
        )
        final_meta_dataset = concat_meta_dataset.filter(
            lambda dp: len(dp['cleaned_metadata']) > 30,
            num_proc=NUM_WORKERS
        )
        for item_id, cleaned_meta in zip(
            final_meta_dataset['parent_asin'],
            final_meta_dataset['cleaned_metadata']
        ):
            all_cleaned_item_metadata[item_id] = cleaned_meta

    print(f'Total items with metadata: {len(all_cleaned_item_metadata)}')
    print("Size of filtered metadata: ", len(final_meta_dataset))
    # Load reviews
    output_review = []
    output_metadata = []
    for category in all_categories:
        # review_dataset = load_dataset(
        #     'McAuley-Lab/Amazon-Reviews-2023',
        #     f'raw_review_{category}',
        #     split='full'
        #     # trust_remote_code=True
        # )

        data_url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"
        df = pd.read_json(data_url, lines=True)
        review_dataset = Dataset.from_pandas(df)

        concat_review_dataset = review_dataset.map(concat_review, num_proc=NUM_WORKERS)
        print("Size of concatenated reviews: ", len(concat_review_dataset))
        
        # Ensure the filter actually sees the global metadata
        final_review_dataset = concat_review_dataset.filter(
            filter_reviews,
            fn_kwargs={"metadata_store": all_cleaned_item_metadata}, # Explicitly pass the store
            num_proc=NUM_WORKERS
        )
        print("Size of filtered reviews: ", len(final_review_dataset))
        
        output_review.extend(final_review_dataset['cleaned_review'])
        valid_metas = [all_cleaned_item_metadata.get(id, "") for id in final_review_dataset['parent_asin']]
        output_metadata.extend(valid_metas)

        # concat_review_dataset = review_dataset.map(
        #     concat_review,
        #     num_proc=NUM_WORKERS
        # )
        # final_review_dataset = concat_review_dataset.filter(
        #     filter_reviews,
        #     num_proc=NUM_WORKERS
        # )
        # output_review.extend(final_review_dataset['cleaned_review'])
        # output_metadata.extend(
        #     [all_cleaned_item_metadata[_] for _ in final_review_dataset['parent_asin']]
        # )

    # Save pretraining data
    df = pd.DataFrame({
        'review': output_review,
        'meta': output_metadata
    })
    print(f'Total samples: {len(df)}')
    df.to_csv('clean_review_meta.tsv', sep='\t', lineterminator='\n', index=False)
