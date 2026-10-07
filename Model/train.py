"""
Train the author similarity model.

Usage:
    python Model/train.py                              # default crawl_data.json
    python Model/train.py --data my_data.json          # custom data file
    python Model/train.py --data my_data.json --neg 5  # more negatives
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .data_prep import create_training_data_from_file
from .model import AuthorSimilarityModel
from .features import get_feature_names


def parse_training_arguments():
    parser = argparse.ArgumentParser(description="Train author similarity model")
    parser.add_argument("--data", default="crawl_data.json", help="Path to crawl data JSON")
    parser.add_argument("--neg", type=int, default=3, help="Negative pair ratio (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lda-topics", type=int, default=20, help="Number of LDA topics (default: 20)")
    return parser.parse_args()


def main():
    args = parse_training_arguments()
    
    X, y, pair_info, lda_model, lda_vec, board_vocab = create_training_data_from_file(
        filepath=args.data,
        neg_ratio=args.neg,
        seed=args.seed,
        lda_topics=args.lda_topics
    )
    
    model = AuthorSimilarityModel()
    feature_names = get_feature_names(lda_topics=args.lda_topics, board_vocab=board_vocab)
    model.train(X, y, feature_names=feature_names, lda_model=lda_model, lda_vectorizer=lda_vec, board_vocab=board_vocab)
    
    model.save()


if __name__ == "__main__":
    main()
