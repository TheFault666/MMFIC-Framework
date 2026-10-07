"""
Author Similarity Model.

A Gradient Boosted classifier that predicts whether two sets of posts
were written by the same person, based on their feature difference vector.

Architecture:
    Input:  |features_A - features_B|  (absolute difference)
    Output: P(same_person)  (0.0 to 1.0)

Why GradientBoosting over deep learning:
    - Works well with small datasets (100s of pairs, not millions)
    - Interpretable (feature importances show what matters)
    - Fast to train and predict
    - No GPU needed
    - Handles the mixed feature types (rates, counts, ratios) naturally
"""

import warnings
import json
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score, average_precision_score
)

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODEL_DIR / "trained_model.joblib"
DEFAULT_SCALER_PATH = MODEL_DIR / "scaler.joblib"
DEFAULT_META_PATH = MODEL_DIR / "model_meta.json"


class AuthorSimilarityModel:
    """Predicts if two forum users are the same person.
    
    Usage:
        # Training
        model = AuthorSimilarityModel()
        model.train(X, y)
        model.save()
        
        # Prediction
        model = AuthorSimilarityModel.load()
        score = model.predict_similarity(posts_a, posts_b)
    """
    
    def __init__(self):
        self.classifier = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_meta = {}
        self.lda_model = None
        self.lda_vectorizer = None
        self.board_vocab = None
    
    @staticmethod
    def _generate_model_version():
        """Generate a model version string based on timestamp."""
        return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _run_cross_validation(self, X_scaled, y):
        """Run 5-fold stratified cross-validation, returning (roc_scores, acc_scores)."""
        cv = StratifiedKFold(n_splits=min(5, sum(y), sum(y==0)), shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            self.classifier, X_scaled, y, 
            cv=cv, scoring="roc_auc"
        )
        cv_acc = cross_val_score(
            self.classifier, X_scaled, y,
            cv=cv, scoring="accuracy"
        )
        return cv_scores, cv_acc
    
    def _build_training_meta(self, X, y, cv_scores, cv_acc, train_auc, train_ap):
        """Assemble the training metadata dict."""
        return {
            "n_pairs": int(len(X)),
            "n_positive": int(sum(y)),
            "n_negative": int(sum(y == 0)),
            "n_features": int(X.shape[1]),
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "cv_accuracy_mean": float(cv_acc.mean()),
            "train_roc_auc": float(train_auc),
            "train_avg_precision": float(train_ap),
            "trained_at": datetime.now().isoformat(),
            "model_version": self._generate_model_version(),
        }
    
    def _log_training_to_db(self):
        """Log a completed training run to the fingerprint store database."""
        from .fingerprint_store import log_training, init_fingerprint_tables
        init_fingerprint_tables()
        log_training(
            metrics=self.training_meta,
            model_version=self.training_meta["model_version"]
        )
    
    def train(self, X, y, feature_names=None, lda_model=None, lda_vectorizer=None, board_vocab=None):
        """Train the model on labeled pairs.
        
        Args:
            X: numpy array of shape (n_pairs, n_features) — feature differences
            y: numpy array of shape (n_pairs,) — 1=same, 0=different
            feature_names: optional list of feature names for interpretability
            lda_model: fitted LatentDirichletAllocation object
            lda_vectorizer: fitted CountVectorizer object
            board_vocab: list of board names
        
        Returns:
            dict with training metrics
        """
        self.lda_model = lda_model
        self.lda_vectorizer = lda_vectorizer
        self.board_vocab = board_vocab
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        X_scaled = self.scaler.fit_transform(X)
        cv_scores, cv_acc = self._run_cross_validation(X_scaled, y)
        
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        y_prob = self.classifier.predict_proba(X_scaled)[:, 1]
        
        train_auc = roc_auc_score(y, y_prob)
        train_ap = average_precision_score(y, y_prob)
        
        self.training_meta = self._build_training_meta(X, y, cv_scores, cv_acc, train_auc, train_ap)
        self._log_training_to_db()
        
        return self.training_meta
    
    def get_training_stats(self):
        """Get stats about what data this model was trained on."""
        if not self.training_meta:
            return "Model not trained yet."
        
        m = self.training_meta
        return (
            f"Model version: {m.get('model_version', 'unknown')}\n"
            f"Trained at:    {m.get('trained_at', 'unknown')}\n"
            f"Training data: {m['n_pairs']} pairs "
            f"({m['n_positive']} positive, {m['n_negative']} negative)\n"
            f"Features:      {m['n_features']}\n"
            f"CV ROC-AUC:    {m['cv_roc_auc_mean']:.4f} (+/- {m['cv_roc_auc_std']:.4f})\n"
            f"CV Accuracy:   {m['cv_accuracy_mean']:.4f}\n"
            f"Train ROC-AUC: {m['train_roc_auc']:.4f}"
        )
    
    def _check_model_is_trained(self):
        """Check if the model has been trained or loaded."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
    
    def predict_pair(self, feature_diff):
        """Predict similarity score for a prepared feature difference vector.
        
        Args:
            feature_diff: numpy array of shape (1, n_features)
        
        Returns:
            float: probability (0.0-1.0) that the pair is the same person
        """
        self._check_model_is_trained()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
            X_scaled = self.scaler.transform(feature_diff)
            prob = self.classifier.predict_proba(X_scaled)[:, 1]
            
        return float(prob[0])
    
    def _evaluate_tie_breaker(self, score, feature_diff, cosine_tie_breaker):
        """Evaluate tie breaking logic and return a verdict and reason."""
        if feature_diff.shape[1] >= 272:
            tie_breaker_diff = feature_diff[0, 238:272]
            magnitude = np.linalg.norm(tie_breaker_diff)
            
            if magnitude < (1.0 - cosine_tie_breaker) * 5.0 and score > 0.5:
                return "MATCH", f"Promoted by tie-breaker (magnitude: {magnitude:.3f})"
            if magnitude > 2.0:
                return "MISMATCH", f"Demoted by tie-breaker (magnitude: {magnitude:.3f})"
        return "UNCERTAIN", "Remains uncertain after tie-breaker"

    def _resolve_twd_verdict(self, score, feature_diff, match_threshold, mismatch_threshold, cosine_tie_breaker):
        """Apply Three-Way Decision logic to a raw similarity score.
        
        Returns (score, verdict, metadata) without modifying any state.
        """
        if score >= match_threshold:
            return score, "MATCH", {"reason": "Confident match by LightGBM"}
        if score <= mismatch_threshold:
            return score, "MISMATCH", {"reason": "Confident mismatch by LightGBM"}
        
        verdict, reason = self._evaluate_tie_breaker(score, feature_diff, cosine_tie_breaker)
        return score, verdict, {"reason": reason}
    
    def predict_pair_twd(self, feature_diff, match_threshold=0.8, mismatch_threshold=0.3, cosine_tie_breaker=0.85):
        """Three-Way Decision prediction logic.
        
        Returns:
            (score, verdict, metadata)
        """
        score = self.predict_pair(feature_diff)
        return self._resolve_twd_verdict(score, feature_diff, match_threshold, mismatch_threshold, cosine_tie_breaker)
    
    def predict_similarity(self, posts_a, posts_b, timestamps_a=None, timestamps_b=None, sources_a=None, sources_b=None, match_threshold=0.8, mismatch_threshold=0.3, cosine_tie_breaker=0.85):
        """Predict if two users are the same person from their posts.
        
        Args:
            posts_a: list of posts from user A
            posts_b: list of posts from user B
        
        Returns:
            (score, verdict, meta)
        """
        from .data_prep import prepare_prediction_pair
        
        diff = prepare_prediction_pair(
            posts_a, posts_b, timestamps_a, timestamps_b, 
            sources_a, sources_b, self.lda_model, self.lda_vectorizer, self.board_vocab
        )
        return self.predict_pair_twd(diff, match_threshold, mismatch_threshold, cosine_tie_breaker)
    
    def batch_compare(self, profiles, match_threshold=0.8, mismatch_threshold=0.3, cosine_tie_breaker=0.85):
        """Compare all user pairs and return a ranked similarity matrix with Three-Way verdicts.
        
        Args:
            profiles: dict {username: {"posts": [...], "timestamps": [...], "sources": [...]}}
        
        Returns:
            list of (user_a, user_b, score, verdict) tuples, sorted by score descending
        """
        from .features import extract_user_features
        
        users = list(profiles.keys())
        features = {}
        
        for user in users:
            feat = extract_user_features(
                profiles[user].get("posts", []),
                profiles[user].get("timestamps", []),
                profiles[user].get("sources", []),
                self.lda_model,
                self.lda_vectorizer,
                self.board_vocab
            )
            features[user] = feat
        
        results = []
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user_a, user_b = users[i], users[j]
                diff = np.abs(features[user_a] - features[user_b]).reshape(1, -1)
                diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                
                score, verdict, _ = self.predict_pair_twd(diff, match_threshold, mismatch_threshold, cosine_tie_breaker)
                results.append((user_a, user_b, score, verdict))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def save(self, model_path=None, scaler_path=None, meta_path=None):
        """Save trained model to disk."""
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        scaler_path = Path(scaler_path or DEFAULT_SCALER_PATH)
        meta_path = Path(meta_path or DEFAULT_META_PATH)
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        if self.lda_model:
            joblib.dump(self.lda_model, model_path.with_name("lda_model.joblib"))
        if self.lda_vectorizer:
            joblib.dump(self.lda_vectorizer, model_path.with_name("lda_vectorizer.joblib"))
        if self.board_vocab:
            joblib.dump(self.board_vocab, model_path.with_name("board_vocab.joblib"))
        
        with open(meta_path, "w") as f:
            json.dump(self.training_meta, f, indent=2)
    
    @classmethod
    def load(cls, model_path=None, scaler_path=None, meta_path=None):
        """Load a trained model from disk."""
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        scaler_path = Path(scaler_path or DEFAULT_SCALER_PATH)
        meta_path = Path(meta_path or DEFAULT_META_PATH)
        
        instance = cls()
        instance.classifier = joblib.load(model_path)
        instance.scaler = joblib.load(scaler_path)
        instance.is_trained = True
        
        lda_model_path = model_path.with_name("lda_model.joblib")
        if lda_model_path.exists():
            instance.lda_model = joblib.load(lda_model_path)
            
        lda_vec_path = model_path.with_name("lda_vectorizer.joblib")
        if lda_vec_path.exists():
            instance.lda_vectorizer = joblib.load(lda_vec_path)
            
        board_vocab_path = model_path.with_name("board_vocab.joblib")
        if board_vocab_path.exists():
            instance.board_vocab = joblib.load(board_vocab_path)
        
        if meta_path.exists():
            with open(meta_path) as f:
                instance.training_meta = json.load(f)
        
        return instance
