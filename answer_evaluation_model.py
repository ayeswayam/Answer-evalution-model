import os
import re
import numpy as np
import pickle
import nltk
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class AnswerEvaluationSystem:
    def __init__(self):
        # Ensure NLTK resources are downloaded
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            print(f"NLTK download warning: {e}")

        # Initialize TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Load stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Stop words loading warning: {e}")
            self.stop_words = set()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by:
        1. Converting to lowercase
        2. Removing special characters
        3. Removing stopwords
        4. Tokenizing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

    def calculate_semantic_similarity(self, student_answer: str, model_answer: str) -> float:
        """
        Calculate semantic similarity using TF-IDF and cosine similarity
        """
        # Preprocess both answers
        processed_student = self.preprocess_text(student_answer)
        processed_model = self.preprocess_text(model_answer)
        
        # Create TF-IDF vectors
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_student, processed_model])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return cosine_sim

    def evaluate_grammar(self, text: str) -> float:
        """
        Basic grammar evaluation using simple heuristics
        """
        # Check sentence structure
        sentences = nltk.sent_tokenize(text)
        
        # Calculate complexity metrics
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
        
        # Simple scoring based on sentence complexity and structure
        if avg_sentence_length < 5:
            return 0.4
        elif avg_sentence_length < 10:
            return 0.6
        elif avg_sentence_length < 15:
            return 0.8
        else:
            return 1.0

    def keyword_match(self, student_answer: str, model_answer: str) -> float:
        """
        Calculate keyword matching score
        """
        # Preprocess and tokenize
        student_tokens = set(self.preprocess_text(student_answer).split())
        model_tokens = set(self.preprocess_text(model_answer).split())
        
        # Calculate keyword overlap
        keyword_overlap = len(student_tokens.intersection(model_tokens))
        total_keywords = len(model_tokens)
        
        # Prevent division by zero
        if total_keywords == 0:
            return 0
        
        return min(keyword_overlap / total_keywords, 1.0)

    def generate_feedback(self, semantic_score: float, keyword_score: float, grammar_score: float) -> str:
        """
        Generate constructive feedback based on scores
        """
        feedback_parts = []
        
        if semantic_score < 0.3:
            feedback_parts.append("Your answer seems to deviate from the expected content. Try to address the key points more directly.")
        elif semantic_score < 0.6:
            feedback_parts.append("Your answer partially captures the main ideas. Consider expanding on the key concepts.")
        else:
            feedback_parts.append("Excellent understanding of the content. Your answer aligns very well with the expected response.")
        
        if keyword_score < 0.4:
            feedback_parts.append("Try to include more key technical terms and specific vocabulary related to the topic.")
        
        if grammar_score < 0.6:
            feedback_parts.append("Work on improving sentence structure and clarity of expression.")
        
        return " ".join(feedback_parts)

def evaluate_single_answer(student_answer: str, model_answer: str) -> Dict[str, Any]:
    """
    Comprehensive answer evaluation function
    """
    # Initialize the evaluation system
    evaluator = AnswerEvaluationSystem()
    
    # Calculate individual scores
    semantic_score = evaluator.calculate_semantic_similarity(student_answer, model_answer)
    keyword_score = evaluator.keyword_match(student_answer, model_answer)
    grammar_score = evaluator.evaluate_grammar(student_answer)
    
    # Calculate overall score (weighted average)
    overall_score = (
        0.4 * semantic_score + 
        0.3 * keyword_score + 
        0.3 * grammar_score
    )
    
    # Generate feedback
    feedback = evaluator.generate_feedback(semantic_score, keyword_score, grammar_score)
    
    return {
        'overall_score': round(overall_score * 100, 2),
        'semantic_score': round(semantic_score * 100, 2),
        'keyword_score': round(keyword_score * 100, 2),
        'grammar_score': round(grammar_score * 100, 2),
        'structure_score': round(grammar_score * 100, 2),
        'feedback': feedback
    }

def evaluate_answers(student_answers, correct_answers, evaluation_system=None):
    """
    Batch evaluation of answers
    """
    if evaluation_system is None:
        evaluation_system = AnswerEvaluationSystem()
    
    results = []
    for student_answer, correct_answer in zip(student_answers, correct_answers):
        results.append(evaluate_single_answer(student_answer, correct_answer))
    
    return results
