"""
Model package for author similarity detection.

Uses stylometric, lexical, and behavioral features to determine
if two forum usernames could be the same person.

Usage:
    from Model.features import extract_user_features
    from Model.model import AuthorSimilarityModel
    from Model.data_prep import prepare_training_pairs, load_crawl_data
"""