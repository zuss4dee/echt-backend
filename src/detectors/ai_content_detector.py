"""
AI Content Detector Module

Detects AI-generated content through linguistic analysis.
"""

import re
import math
from collections import Counter


class AIContentDetector:
    """Detects AI-generated content through text analysis."""
    
    # Common AI-typical phrases and patterns
    AI_TYPICAL_PHRASES = [
        'i hope this', 'i wanted to', 'i am writing', 'please find',
        'i would like to', 'i am pleased to', 'i trust this',
        'should you have', 'please do not hesitate', 'i look forward',
        'best regards', 'kind regards', 'sincerely', 'yours sincerely',
        'furthermore', 'moreover', 'additionally', 'in conclusion',
        'it is important to note', 'it should be noted', 'it is worth noting'
    ]
    
    def analyze(self, text):
        """
        Analyze text for AI-generated content indicators.
        
        Args:
            text: str, document text to analyze
            
        Returns:
            dict: {
                'is_ai_generated': bool,
                'confidence': float (0-100),
                'indicators': list of str,
                'metrics': dict with detailed metrics
            }
        """
        if not text or len(text.strip()) < 50:
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'indicators': ['Insufficient text for analysis'],
                'metrics': {}
            }
        
        metrics = {}
        indicators = []
        confidence_score = 0.0
        
        # Calculate punctuation diversity
        punctuation_diversity = self._calculate_punctuation_diversity(text)
        metrics['punctuation_diversity'] = punctuation_diversity
        if punctuation_diversity < 0.05:
            indicators.append(f'Low punctuation diversity ({punctuation_diversity:.2f}) - may indicate AI generation')
            confidence_score += 15
        
        # Calculate word entropy
        word_entropy = self._calculate_word_entropy(text)
        metrics['word_entropy'] = word_entropy
        if word_entropy < 8.5:
            indicators.append(f'Low word entropy ({word_entropy:.2f}) - text may be AI-generated')
            confidence_score += 20
        
        # Calculate paragraph uniformity
        paragraph_uniformity = self._calculate_paragraph_uniformity(text)
        metrics['paragraph_uniformity'] = paragraph_uniformity
        if paragraph_uniformity > 0.85:
            indicators.append(f'High paragraph uniformity ({paragraph_uniformity:.2f}) - may indicate AI generation')
            confidence_score += 15
        
        # Detect AI-typical phrases
        ai_phrases_found, phrase_locations = self._detect_ai_phrases(text)
        metrics['ai_phrases_count'] = len(ai_phrases_found)
        metrics['ai_phrases'] = ai_phrases_found
        metrics['ai_phrase_locations'] = phrase_locations  # Store locations for display
        if len(ai_phrases_found) >= 2:
            indicators.append(f'AI-typical phrases detected: {len(ai_phrases_found)} found')
            confidence_score += 10 * min(len(ai_phrases_found), 5)  # Cap at 5 phrases
        
        # Calculate sentence length variance
        sentence_variance = self._calculate_sentence_variance(text)
        metrics['sentence_variance'] = sentence_variance
        if sentence_variance < 50:  # Low variance indicates uniform sentence lengths
            indicators.append('Low sentence length variance - may indicate AI generation')
            confidence_score += 10
        
        # Check for repetitive patterns
        repetition_score = self._check_repetition(text)
        metrics['repetition_score'] = repetition_score
        if repetition_score > 0.3:
            indicators.append('High text repetition detected - may indicate AI generation')
            confidence_score += 15
        
        # Normalize confidence to 0-100
        confidence_score = min(confidence_score, 100)
        
        is_ai_generated = confidence_score >= 30  # Threshold for AI detection
        
        return {
            'is_ai_generated': is_ai_generated,
            'confidence': confidence_score,
            'indicators': indicators,
            'metrics': metrics
        }
    
    def _calculate_punctuation_diversity(self, text):
        """Calculate punctuation diversity (ratio of unique punctuation to total punctuation)."""
        # Extract all punctuation
        punctuation = re.findall(r'[^\w\s]', text)
        if not punctuation:
            return 0.0
        
        unique_punct = len(set(punctuation))
        total_punct = len(punctuation)
        
        return unique_punct / total_punct if total_punct > 0 else 0.0
    
    def _calculate_word_entropy(self, text):
        """Calculate Shannon entropy of words (measure of randomness/variety)."""
        # Extract words (lowercase)
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        # Count word frequencies
        word_counts = Counter(words)
        total_words = len(words)
        
        # Calculate entropy
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_paragraph_uniformity(self, text):
        """Calculate paragraph length uniformity (higher = more uniform = more AI-like)."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            return 0.0
        
        # Calculate paragraph lengths
        para_lengths = [len(p) for p in paragraphs]
        if not para_lengths:
            return 0.0
        
        # Calculate coefficient of variation (std dev / mean)
        mean_length = sum(para_lengths) / len(para_lengths)
        if mean_length == 0:
            return 0.0
        
        variance = sum((x - mean_length) ** 2 for x in para_lengths) / len(para_lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_length if mean_length > 0 else 0
        
        # Uniformity is inverse of coefficient of variation
        # Normalize to 0-1 range (assuming CV typically 0-2)
        uniformity = 1.0 / (1.0 + cv)
        
        return uniformity
    
    def _detect_ai_phrases(self, text):
        """Detect AI-typical phrases in text and return with locations."""
        text_lower = text.lower()
        found_phrases = []
        phrase_locations = []  # List of dicts with phrase, start_pos, end_pos, context
        
        for phrase in self.AI_TYPICAL_PHRASES:
            # Find all occurrences of the phrase
            start = 0
            while True:
                pos = text_lower.find(phrase, start)
                if pos == -1:
                    break
                
                # Get context (50 chars before and after)
                context_start = max(0, pos - 50)
                context_end = min(len(text), pos + len(phrase) + 50)
                context = text[context_start:context_end]
                
                # Find the actual phrase in original case
                phrase_original = text[pos:pos + len(phrase)]
                
                found_phrases.append(phrase_original)
                phrase_locations.append({
                    'phrase': phrase_original,
                    'position': pos,
                    'end_position': pos + len(phrase),
                    'context': context,
                    'context_start': context_start,
                    'context_end': context_end
                })
                
                start = pos + 1
        
        return found_phrases, phrase_locations
    
    def _calculate_sentence_variance(self, text):
        """Calculate variance in sentence lengths."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        sentence_lengths = [len(s) for s in sentences]
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        
        if mean_length == 0:
            return 0.0
        
        variance = sum((x - mean_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        return variance
    
    def _check_repetition(self, text):
        """Check for repetitive text patterns."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 10:
            return 0.0
        
        # Check for n-gram repetition (bigrams and trigrams)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        # Calculate repetition ratio
        total_bigrams = len(bigrams)
        unique_bigrams = len(set(bigrams))
        bigram_repetition = 1.0 - (unique_bigrams / total_bigrams) if total_bigrams > 0 else 0.0
        
        total_trigrams = len(trigrams)
        unique_trigrams = len(set(trigrams))
        trigram_repetition = 1.0 - (unique_trigrams / total_trigrams) if total_trigrams > 0 else 0.0
        
        # Average repetition score
        repetition_score = (bigram_repetition + trigram_repetition) / 2.0
        
        return repetition_score
