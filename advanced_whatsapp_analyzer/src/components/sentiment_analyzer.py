"""
Advanced Sentiment Analysis Module
Provides multi-model sentiment analysis with emotion detection, 
toxicity analysis, and confidence scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# NLP Libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    import torch
except ImportError as e:
    logging.warning(f"Some sentiment analysis libraries not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Individual sentiment analysis result"""
    text: str
    positive: float
    negative: float
    neutral: float
    compound: float
    label: str  # 'positive', 'negative', 'neutral'
    confidence: float
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    toxicity_score: Optional[float] = None
    is_toxic: bool = False

@dataclass
class BulkSentimentResult:
    """Bulk sentiment analysis results"""
    results: List[SentimentResult]
    summary: Dict[str, Any]
    processing_time: float
    model_used: str

class MultiModelSentimentAnalyzer:
    """Advanced sentiment analyzer with multiple models"""
    
    def __init__(self, model_type: str = "vader", language: str = "en"):
        self.model_type = model_type
        self.language = language
        self.models = {}
        self.emotion_model = None
        self.toxicity_model = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            # VADER Sentiment
            if self.model_type in ["vader", "multi"]:
                self.models["vader"] = SentimentIntensityAnalyzer()
                logger.info("✅ VADER sentiment model initialized")
            
            # TextBlob
            if self.model_type in ["textblob", "multi"]:
                self.models["textblob"] = "textblob"  # TextBlob doesn't need initialization
                logger.info("✅ TextBlob sentiment model initialized")
            
            # Transformer models
            if self.model_type in ["transformers", "multi"]:
                try:
                    # Use a lightweight sentiment model
                    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                    self.models["transformers"] = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        return_all_scores=True,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("✅ Transformers sentiment model initialized")
                except Exception as e:
                    logger.warning(f"Failed to load transformers model: {e}")
            
            # Emotion detection model
            try:
                self.emotion_model = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("✅ Emotion detection model initialized")
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
            
            # Toxicity detection model  
            try:
                self.toxicity_model = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("✅ Toxicity detection model initialized")
            except Exception as e:
                logger.warning(f"Failed to load toxicity model: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
            # Fallback to VADER only
            self.models["vader"] = SentimentIntensityAnalyzer()
            self.model_type = "vader"
    
    def analyze_single(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text"""
        if not text or pd.isna(text) or len(str(text).strip()) == 0:
            return SentimentResult(
                text="", positive=0.0, negative=0.0, neutral=1.0,
                compound=0.0, label="neutral", confidence=0.0
            )
        
        text = str(text).strip()
        results = {}
        
        # VADER Analysis
        if "vader" in self.models:
            vader_result = self.models["vader"].polarity_scores(text)
            results["vader"] = {
                "positive": vader_result["pos"],
                "negative": vader_result["neg"], 
                "neutral": vader_result["neu"],
                "compound": vader_result["compound"]
            }
        
        # TextBlob Analysis
        if "textblob" in self.models:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # Convert to positive/negative/neutral scale
                if polarity > 0.1:
                    positive = (polarity + 1) / 2
                    negative = 0
                    neutral = 1 - positive
                elif polarity < -0.1:
                    negative = abs(polarity)
                    positive = 0
                    neutral = 1 - negative
                else:
                    neutral = 1
                    positive = negative = 0
                
                results["textblob"] = {
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "compound": polarity
                }
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")
        
        # Transformers Analysis
        if "transformers" in self.models:
            try:
                transformer_result = self.models["transformers"](text)[0]
                
                # Convert to our format
                scores = {item["label"].lower(): item["score"] for item in transformer_result}
                compound = scores.get("positive", 0) - scores.get("negative", 0)
                
                results["transformers"] = {
                    "positive": scores.get("positive", 0),
                    "negative": scores.get("negative", 0),
                    "neutral": scores.get("neutral", 0),
                    "compound": compound
                }
            except Exception as e:
                logger.warning(f"Transformers analysis failed: {e}")
        
        # Combine results (weighted average)
        if self.model_type == "multi" and len(results) > 1:
            combined = self._combine_results(results)
        else:
            # Use single model result
            model_name = list(results.keys())[0]
            combined = results[model_name]
        
        # Determine label and confidence
        label, confidence = self._determine_label_and_confidence(combined)
        
        # Get emotion if available
        emotion, emotion_confidence = self._analyze_emotion(text)
        
        # Get toxicity score if available
        toxicity_score, is_toxic = self._analyze_toxicity(text)
        
        return SentimentResult(
            text=text,
            positive=combined["positive"],
            negative=combined["negative"],
            neutral=combined["neutral"],
            compound=combined["compound"],
            label=label,
            confidence=confidence,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            toxicity_score=toxicity_score,
            is_toxic=is_toxic
        )
    
    def _combine_results(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine multiple model results with weighted average"""
        weights = {
            "vader": 0.3,
            "textblob": 0.2,
            "transformers": 0.5
        }
        
        combined = {"positive": 0, "negative": 0, "neutral": 0, "compound": 0}
        total_weight = 0
        
        for model, result in results.items():
            weight = weights.get(model, 0.33)
            total_weight += weight
            
            for metric in combined:
                combined[metric] += result[metric] * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for metric in combined:
                combined[metric] /= total_weight
        
        return combined
    
    def _determine_label_and_confidence(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """Determine sentiment label and confidence from scores"""
        compound = scores["compound"]
        positive = scores["positive"]
        negative = scores["negative"]
        neutral = scores["neutral"]
        
        if compound >= 0.05:
            label = "positive"
            confidence = positive
        elif compound <= -0.05:
            label = "negative"
            confidence = negative
        else:
            label = "neutral"
            confidence = neutral
        
        return label, min(confidence, 1.0)
    
    def _analyze_emotion(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Analyze emotion in text"""
        if not self.emotion_model:
            return None, None
        
        try:
            if len(text.strip()) < 3:  # Skip very short texts
                return None, None
                
            emotion_result = self.emotion_model(text)[0]
            
            # Get top emotion
            top_emotion = max(emotion_result, key=lambda x: x["score"])
            
            return top_emotion["label"].lower(), top_emotion["score"]
            
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return None, None
    
    def _analyze_toxicity(self, text: str) -> Tuple[Optional[float], bool]:
        """Analyze toxicity in text"""
        if not self.toxicity_model:
            return None, False
        
        try:
            if len(text.strip()) < 3:  # Skip very short texts
                return None, False
                
            toxicity_result = self.toxicity_model(text)
            
            # Extract toxicity score
            if isinstance(toxicity_result, list) and len(toxicity_result) > 0:
                result = toxicity_result[0]
                if result["label"] == "TOXIC":
                    toxicity_score = result["score"]
                    is_toxic = toxicity_score > 0.5
                else:
                    toxicity_score = 1 - result["score"]
                    is_toxic = False
            else:
                return None, False
            
            return toxicity_score, is_toxic
            
        except Exception as e:
            logger.warning(f"Toxicity analysis failed: {e}")
            return None, False
    
    def analyze_bulk(self, texts: List[str], batch_size: int = 100, 
                    use_threading: bool = True) -> BulkSentimentResult:
        """Analyze sentiment for multiple texts"""
        import time
        start_time = time.time()
        
        if not texts:
            return BulkSentimentResult(
                results=[], summary={}, processing_time=0.0, model_used=self.model_type
            )
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and len(str(text).strip()) > 0]
        
        if use_threading and len(valid_texts) > batch_size:
            # Use threading for large datasets
            results = self._analyze_with_threading(valid_texts, batch_size)
        else:
            # Sequential processing
            results = []
            for i, text in valid_texts:
                result = self.analyze_single(text)
                results.append((i, result))
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        sentiment_results = [result for _, result in results]
        
        # Generate summary
        summary = self._generate_summary(sentiment_results)
        
        processing_time = time.time() - start_time
        
        return BulkSentimentResult(
            results=sentiment_results,
            summary=summary,
            processing_time=processing_time,
            model_used=self.model_type
        )
    
    def _analyze_with_threading(self, indexed_texts: List[Tuple[int, str]], 
                              batch_size: int) -> List[Tuple[int, SentimentResult]]:
        """Analyze texts using threading for better performance"""
        results = []
        
        def process_batch(batch):
            batch_results = []
            for i, text in batch:
                result = self.analyze_single(text)
                batch_results.append((i, result))
            return batch_results
        
        # Create batches
        batches = [
            indexed_texts[i:i + batch_size] 
            for i in range(0, len(indexed_texts), batch_size)
        ]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            batch_results = list(executor.map(process_batch, batches))
        
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    def _generate_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Generate summary statistics from sentiment results"""
        if not results:
            return {}
        
        # Basic sentiment distribution
        labels = [r.label for r in results]
        sentiment_dist = {
            "positive": labels.count("positive"),
            "negative": labels.count("negative"), 
            "neutral": labels.count("neutral")
        }
        
        # Percentages
        total = len(results)
        sentiment_pct = {
            f"{k}_pct": round(v / total * 100, 2) if total > 0 else 0
            for k, v in sentiment_dist.items()
        }
        
        # Average scores
        avg_scores = {
            "avg_positive": np.mean([r.positive for r in results]),
            "avg_negative": np.mean([r.negative for r in results]),
            "avg_neutral": np.mean([r.neutral for r in results]),
            "avg_compound": np.mean([r.compound for r in results]),
            "avg_confidence": np.mean([r.confidence for r in results])
        }
        
        # Emotion analysis summary
        emotions = [r.emotion for r in results if r.emotion]
        emotion_dist = {}
        if emotions:
            from collections import Counter
            emotion_counts = Counter(emotions)
            emotion_dist = dict(emotion_counts.most_common(10))
        
        # Toxicity summary
        toxic_results = [r for r in results if r.toxicity_score is not None]
        toxicity_summary = {}
        if toxic_results:
            toxicity_summary = {
                "toxic_count": sum(1 for r in toxic_results if r.is_toxic),
                "avg_toxicity_score": np.mean([r.toxicity_score for r in toxic_results]),
                "max_toxicity_score": max([r.toxicity_score for r in toxic_results])
            }
        
        # Time-based patterns (if timestamps available)
        summary = {
            "total_messages": total,
            "sentiment_distribution": sentiment_dist,
            "sentiment_percentages": sentiment_pct,
            "average_scores": avg_scores,
            "emotion_distribution": emotion_dist,
            "toxicity_summary": toxicity_summary,
            "overall_sentiment": max(sentiment_dist, key=sentiment_dist.get),
            "sentiment_score": avg_scores["avg_compound"]
        }
        
        return summary

class SentimentAnalysisEngine:
    """High-level sentiment analysis engine for DataFrames"""
    
    def __init__(self, model_type: str = "vader", language: str = "en"):
        self.analyzer = MultiModelSentimentAnalyzer(model_type, language)
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "message",
                         user_column: str = "user", date_column: str = "date") -> pd.DataFrame:
        """Analyze sentiment for entire DataFrame"""
        logger.info("Starting sentiment analysis for DataFrame...")
        
        # Filter text messages only
        text_mask = df['message_type'] == 'text'
        text_df = df[text_mask].copy()
        
        if text_df.empty:
            logger.warning("No text messages found for sentiment analysis")
            return df
        
        # Extract texts
        texts = text_df[text_column].astype(str).tolist()
        
        # Perform bulk analysis
        bulk_result = self.analyzer.analyze_bulk(texts)
        
        # Add results to DataFrame
        sentiment_df = pd.DataFrame([
            {
                'positive_score': r.positive,
                'negative_score': r.negative,
                'neutral_score': r.neutral,
                'compound_score': r.compound,
                'sentiment_label': r.label,
                'sentiment_confidence': r.confidence,
                'emotion': r.emotion,
                'emotion_confidence': r.emotion_confidence,
                'toxicity_score': r.toxicity_score,
                'is_toxic': r.is_toxic
            }
            for r in bulk_result.results
        ])
        
        # Merge with original DataFrame
        text_df = text_df.reset_index()
        sentiment_df['original_index'] = text_df['index']
        
        # Initialize sentiment columns for all messages
        sentiment_columns = [
            'positive_score', 'negative_score', 'neutral_score', 'compound_score',
            'sentiment_label', 'sentiment_confidence', 'emotion', 'emotion_confidence',
            'toxicity_score', 'is_toxic'
        ]
        
        for col in sentiment_columns:
            if col not in df.columns:
                if col in ['positive_score', 'negative_score', 'neutral_score', 'compound_score', 'sentiment_confidence', 'emotion_confidence', 'toxicity_score']:
                    df[col] = 0.0
                elif col == 'is_toxic':
                    df[col] = False
                else:
                    df[col] = None
        
        # Update sentiment data for text messages
        for i, row in sentiment_df.iterrows():
            orig_idx = row['original_index']
            for col in sentiment_columns:
                df.loc[orig_idx, col] = row[col]
        
        logger.info(f"✅ Sentiment analysis completed for {len(bulk_result.results)} messages")
        logger.info(f"📊 Summary: {bulk_result.summary['sentiment_distribution']}")
        
        return df
    
    def get_sentiment_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed sentiment insights"""
        if 'sentiment_label' not in df.columns:
            return {"error": "Sentiment analysis not performed yet"}
        
        # Filter text messages with sentiment data
        sentiment_data = df[df['sentiment_label'].notna()]
        
        if sentiment_data.empty:
            return {"error": "No sentiment data available"}
        
        insights = {}
        
        # Overall sentiment distribution
        sentiment_dist = sentiment_data['sentiment_label'].value_counts()
        insights['overall_distribution'] = sentiment_dist.to_dict()
        
        # User-wise sentiment
        if 'user' in df.columns:
            user_sentiment = sentiment_data.groupby('user')['sentiment_label'].value_counts().unstack(fill_value=0)
            insights['user_sentiment'] = user_sentiment.to_dict('index')
        
        # Time-based sentiment trends
        if 'date' in df.columns:
            daily_sentiment = sentiment_data.groupby([sentiment_data['date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)
            insights['daily_trends'] = daily_sentiment.to_dict('index')
        
        # Emotion analysis
        if 'emotion' in df.columns:
            emotion_dist = sentiment_data['emotion'].value_counts()
            insights['emotion_distribution'] = emotion_dist.to_dict()
        
        # Toxicity insights
        if 'is_toxic' in df.columns:
            toxic_count = sentiment_data['is_toxic'].sum()
            insights['toxicity'] = {
                'toxic_messages': int(toxic_count),
                'toxic_percentage': round(toxic_count / len(sentiment_data) * 100, 2)
            }
        
        # Statistical summary
        insights['statistics'] = {
            'avg_compound_score': sentiment_data['compound_score'].mean(),
            'most_positive_day': None,
            'most_negative_day': None
        }
        
        # Find most positive/negative days
        if 'date' in df.columns:
            daily_compound = sentiment_data.groupby(sentiment_data['date'].dt.date)['compound_score'].mean()
            insights['statistics']['most_positive_day'] = str(daily_compound.idxmax())
            insights['statistics']['most_negative_day'] = str(daily_compound.idxmin())
        
        return insights

# Utility functions for easy access
def analyze_sentiment(texts: List[str], model_type: str = "vader") -> List[SentimentResult]:
    """Quick sentiment analysis for list of texts"""
    analyzer = MultiModelSentimentAnalyzer(model_type)
    results = analyzer.analyze_bulk(texts)
    return results.results

def get_sentiment_summary(texts: List[str], model_type: str = "vader") -> Dict[str, Any]:
    """Quick sentiment summary for list of texts"""
    analyzer = MultiModelSentimentAnalyzer(model_type)
    results = analyzer.analyze_bulk(texts)
    return results.summary