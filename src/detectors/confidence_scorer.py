"""
Confidence Scorer Module

Combines all fraud detection indicators to calculate high-confidence scores.
"""


class ConfidenceScorer:
    """Calculates confidence scores based on multiple fraud indicators."""
    
    # Strong fraud indicators (high confidence triggers)
    STRONG_INDICATORS = [
        'suspicious_software', 'timeline_anomaly', 'digital_manipulation',
        'invalid_signatures', 'post_signing_modifications', 'xref_anomalies',
        'no_text_layer', 'ocr_artifacts', 'font_embedding_issues',
        'ela_anomalies', 'smoothing_detected'
    ]
    
    # Weak indicators (common in legitimate documents, only contribute to moderate suspicion)
    WEAK_INDICATORS = [
        'incremental_updates', 'no_signatures', 'object_count',
        'version_inconsistency', 'alignment_issues', 'font_size_issues',
        'margin_issues'
    ]
    
    # Indicator categories and their weights
    INDICATOR_WEIGHTS = {
        'metadata': {
            'suspicious_software': 30,
            'timeline_anomaly': 25,
            'missing_metadata': 15,
            'institutional_indicators': -10  # Reduces suspicion
        },
        'structure': {
            'incremental_updates': 5,  # Reduced - too common
            'xref_anomalies': 25,
            'version_inconsistency': 10  # Reduced
        },
        'content': {
            'font_embedding_issues': 20,
            'system_fonts': 25,
            'font_inconsistency': 15,
            'no_text_layer': 30,
            'ocr_artifacts': 15,
            'alignment_issues': 10,  # Reduced
            'font_size_issues': 8,  # Reduced
            'margin_issues': 8  # Reduced
        },
        'pixel': {
            'smoothing_detected': 20,
            'ela_anomalies': 25
        },
        'signature': {
            'invalid_signatures': 40,
            'timestamp_anomalies': 25,
            'post_signing_modifications': 50
        }
    }
    
    def calculate_confidence(self, all_results):
        """
        Calculate overall confidence score from all detector results.
        
        Args:
            all_results: dict containing results from all detectors:
                - metadata: MetadataDetector results
                - structure: PDFStructureDetector results
                - font: FontDetector results
                - text_layer: TextLayerDetector results
                - layout: LayoutDetector results
                - pixel: PixelDetector results
                - signature: SignatureDetector results
                - embedded: EmbeddedObjectDetector results
                
        Returns:
            dict: {
                'confidence_score': float (0-100),
                'confidence_level': str,
                'fraud_probability': float (0-100),
                'indicator_count': int,
                'indicator_breakdown': dict,
                'recommendation': str
            }
        """
        # Collect all flags and risk scores
        all_flags = []
        category_scores = {}
        indicator_count = 0
        
        # Metadata indicators
        if 'metadata' in all_results:
            metadata_result = all_results['metadata']
            all_flags.extend(metadata_result.get('flags', []))
            category_scores['metadata'] = metadata_result.get('risk_score', 0)
            if metadata_result.get('flags'):
                indicator_count += len(metadata_result['flags'])
        
        # Structure indicators
        if 'structure' in all_results:
            structure_result = all_results['structure']
            all_flags.extend(structure_result.get('flags', []))
            category_scores['structure'] = structure_result.get('risk_score', 0)
            if structure_result.get('flags'):
                indicator_count += len(structure_result['flags'])
        
        # Font indicators
        if 'font' in all_results:
            font_result = all_results['font']
            all_flags.extend(font_result.get('flags', []))
            category_scores['font'] = font_result.get('risk_score', 0)
            if font_result.get('flags'):
                indicator_count += len(font_result['flags'])
        
        # Text layer indicators
        if 'text_layer' in all_results:
            text_result = all_results['text_layer']
            all_flags.extend(text_result.get('flags', []))
            category_scores['text_layer'] = text_result.get('risk_score', 0)
            if text_result.get('flags'):
                indicator_count += len(text_result['flags'])
        
        # Layout indicators
        if 'layout' in all_results:
            layout_result = all_results['layout']
            all_flags.extend(layout_result.get('flags', []))
            category_scores['layout'] = layout_result.get('risk_score', 0)
            if layout_result.get('flags'):
                indicator_count += len(layout_result['flags'])
        
        # Pixel indicators
        if 'noise' in all_results:
            pixel_result = all_results.get('noise', {})
            all_flags.extend(pixel_result.get('flags', []))
            # Pixel risk is already factored into metadata trust score
        
        # Signature indicators
        if 'signature' in all_results:
            signature_result = all_results['signature']
            all_flags.extend(signature_result.get('flags', []))
            category_scores['signature'] = signature_result.get('risk_score', 0)
            if signature_result.get('flags'):
                indicator_count += len(signature_result['flags'])
        
        # Embedded object indicators
        if 'embedded' in all_results:
            embedded_result = all_results['embedded']
            all_flags.extend(embedded_result.get('flags', []))
            category_scores['embedded'] = embedded_result.get('risk_score', 0)
            if embedded_result.get('flags'):
                indicator_count += len(embedded_result['flags'])
        
        # Separate strong and weak indicators
        strong_flags = []
        weak_flags = []
        
        # Strong indicator keywords (actual flag text patterns)
        strong_keywords = [
            'digital manipulation', 'suspicious software', 'timeline anomaly',
            'temporal metadata', 'photoshop', 'gimp', 'canva', 'sejda', 'ilovepdf',
            'invalid signature', 'post signing', 'cross-reference', 'xref',
            'no text layer', 'ocr artifact', 'font embedding', 'smoothing detected',
            'ela anomaly', 'error level analysis'
        ]
        
        # Weak indicator keywords (common in legitimate documents)
        weak_keywords = [
            'incremental', 'object count', 'version inconsistency',
            'alignment', 'font size', 'margin', 'no digital signature'
        ]
        
        for flag in all_flags:
            flag_lower = flag.lower()
            is_strong = any(keyword in flag_lower for keyword in strong_keywords)
            is_weak = any(keyword in flag_lower for keyword in weak_keywords)
            
            if is_strong:
                strong_flags.append(flag)
            elif is_weak:
                weak_flags.append(flag)
            else:
                # Default: check if it's a known strong indicator pattern
                if any(x in flag_lower for x in ['manipulation', 'fraud', 'anomaly', 'suspicious']):
                    strong_flags.append(flag)
                else:
                    # Unknown flags default to weak (conservative approach)
                    weak_flags.append(flag)
        
        strong_count = len(strong_flags)
        weak_count = len(weak_flags)
        
        # Calculate fraud probability (0-100) - only from strong indicators
        total_risk = sum(category_scores.values())
        fraud_probability = min(total_risk, 100)
        
        # Calculate confidence based on STRONG indicators only
        # Weak indicators only contribute to moderate suspicion, never high confidence
        unique_categories = len([c for c in category_scores.values() if c > 0])
        
        # Confidence calculation based on STRONG indicators:
        # - 3+ strong indicators = 90%+ confidence (Definitive Fraud)
        # - 2 strong indicators = 70-89% confidence (High Suspicion)
        # - 1 strong indicator = 50-69% confidence (Moderate Suspicion)
        # - 0 strong indicators but weak indicators = 30-49% confidence (Low Suspicion)
        # - 0 indicators = <30% confidence (Legitimate)
        
        if strong_count >= 3:
            confidence_score = 90 + min(10, (strong_count - 3) * 2)
            confidence_level = 'Definitive Fraud'
            recommendation = 'HIGH CONFIDENCE: Document shows multiple strong fraud indicators. No verification needed.'
        elif strong_count >= 2:
            confidence_score = 70 + min(19, (strong_count - 2) * 5)
            confidence_level = 'High Suspicion'
            recommendation = 'HIGH SUSPICION: Multiple strong fraud indicators detected. Document likely fraudulent.'
        elif strong_count >= 1:
            confidence_score = 50 + min(19, strong_count * 10)
            confidence_level = 'Moderate Suspicion'
            recommendation = 'MODERATE SUSPICION: Strong fraud indicator detected. Recommend verification.'
        elif weak_count >= 2:
            # Only weak indicators - lower confidence
            confidence_score = 30 + min(19, weak_count * 5)
            confidence_level = 'Low Suspicion'
            recommendation = 'LOW SUSPICION: Minor anomalies detected. Document likely legitimate but verify if needed.'
        elif weak_count >= 1:
            confidence_score = 20 + min(9, weak_count * 5)
            confidence_level = 'Low Suspicion'
            recommendation = 'LOW SUSPICION: Minor anomalies detected. Document appears legitimate.'
        else:
            confidence_score = max(0, 100 - fraud_probability)
            confidence_level = 'Low Suspicion'
            recommendation = 'LOW SUSPICION: Minimal fraud indicators. Document appears legitimate.'
        
        # Ensure confidence doesn't exceed 100
        confidence_score = min(confidence_score, 100)
        
        return {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'fraud_probability': fraud_probability,
            'indicator_count': indicator_count,
            'unique_categories': unique_categories,
            'category_scores': category_scores,
            'all_flags': all_flags,
            'recommendation': recommendation,
            'indicator_breakdown': {
                'metadata_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['metadata', 'software', 'timeline', 'author', 'creator'])]),
                'structure_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['incremental', 'xref', 'version', 'structure'])]),
                'content_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['font', 'text', 'layout', 'alignment', 'margin'])]),
                'pixel_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['smoothing', 'noise', 'ela'])]),
                'signature_indicators': len([f for f in all_flags if 'signature' in f.lower()])
            }
        }
