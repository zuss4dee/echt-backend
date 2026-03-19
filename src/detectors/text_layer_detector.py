"""
Text Layer Detector Module

Distinguishes between real text and image-based text (scanned/edited).
"""

import fitz  # PyMuPDF


class TextLayerDetector:
    """Detects fraud indicators through text layer analysis."""
    
    def analyze(self, pdf_doc):
        """
        Analyze text layers for fraud indicators.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            dict: {
                'flags': list of str,
                'findings': dict with detailed findings,
                'risk_score': int (0-100),
                'text_issues': list of detected issues
            }
        """
        flags = []
        findings = {}
        risk_score = 0
        text_issues = []
        
        try:
            # Analyze each page
            page_analyses = []
            total_text_blocks = 0
            total_images = 0
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_analysis = self._analyze_page(page, page_num)
                page_analyses.append(page_analysis)
                total_text_blocks += page_analysis['text_block_count']
                total_images += page_analysis['image_count']
            
            findings['page_analyses'] = page_analyses
            findings['total_text_blocks'] = total_text_blocks
            findings['total_images'] = total_images
            
            # Check if document is image-based (scanned)
            if total_text_blocks == 0 and total_images > 0:
                flags.append('Image-Based Document (No Text Layer) - Likely Scanned')
                risk_score += 30
                text_issues.append('no_text_layer')
                findings['document_type'] = 'scanned_image'
            
            # Check text-to-image ratio
            if total_text_blocks > 0 and total_images > 0:
                ratio = total_images / total_text_blocks
                if ratio > 2.0:  # More images than text blocks
                    flags.append('High Image-to-Text Ratio - Possible Edited Document')
                    risk_score += 20
                    text_issues.append('high_image_ratio')
                    findings['image_text_ratio'] = ratio
            
            # Check for OCR artifacts
            ocr_check = self._check_ocr_artifacts(page_analyses)
            if ocr_check['likely_ocr']:
                flags.append('OCR Artifacts Detected - Document May Have Been Scanned')
                risk_score += 15
                text_issues.append('ocr_artifacts')
                findings['ocr_analysis'] = ocr_check
            
            # Check text positioning consistency
            positioning_check = self._check_text_positioning(page_analyses)
            if positioning_check['inconsistent']:
                flags.append('Text Positioning Inconsistencies')
                risk_score += 15
                text_issues.append('positioning_issues')
                findings['positioning'] = positioning_check
            
            # Check text density
            density_check = self._check_text_density(page_analyses)
            findings['density'] = density_check
            
        except Exception as e:
            flags.append(f'Text Layer Analysis Error: {str(e)}')
            findings['error'] = str(e)
        
        return {
            'flags': flags,
            'findings': findings,
            'risk_score': min(risk_score, 100),
            'text_issues': text_issues
        }
    
    def _analyze_page(self, page, page_num):
        """Analyze a single page for text and images."""
        # Get text blocks
        text_blocks = page.get_text("blocks")
        text_block_count = len(text_blocks)
        
        # Get text as dict for detailed analysis
        text_dict = page.get_text("dict")
        
        # Get images
        images = page.get_images()
        image_count = len(images)
        
        # Calculate text coverage
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        text_area = 0
        for block in text_blocks:
            if len(block) >= 5:  # Block has coordinates
                bbox = block[:4]  # x0, y0, x1, y1
                block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                text_area += block_area
        
        text_coverage = (text_area / page_area * 100) if page_area > 0 else 0
        
        return {
            'page_num': page_num + 1,
            'text_block_count': text_block_count,
            'image_count': image_count,
            'text_coverage': text_coverage,
            'has_text': text_block_count > 0,
            'has_images': image_count > 0,
            'text_dict': text_dict
        }
    
    def _check_ocr_artifacts(self, page_analyses):
        """Check for OCR artifacts that indicate scanning."""
        likely_ocr = False
        indicators = []
        
        for page_analysis in page_analyses:
            text_dict = page_analysis.get('text_dict', {})
            
            # Check for common OCR artifacts
            # 1. Inconsistent character spacing
            # 2. Mixed case issues
            # 3. Character recognition errors
            
            # Get text content
            text_content = text_dict.get('text', '')
            
            # Check for common OCR errors (heuristic)
            if text_content:
                # Check for unusual character patterns
                # OCR often produces certain patterns
                if '|' in text_content and 'l' in text_content:
                    # OCR sometimes confuses | and l
                    indicators.append('character_confusion_patterns')
                
                # Check for spacing issues
                if '  ' in text_content or '\t' in text_content:
                    indicators.append('spacing_issues')
        
        if len(indicators) >= 2:
            likely_ocr = True
        
        return {
            'likely_ocr': likely_ocr,
            'indicators': indicators
        }
    
    def _check_text_positioning(self, page_analyses):
        """Check for text positioning inconsistencies."""
        if len(page_analyses) < 2:
            return {'inconsistent': False}
        
        # Compare text block positions across pages
        inconsistencies = []
        
        # Get average text block positions for first page
        first_page = page_analyses[0]
        if first_page['text_block_count'] == 0:
            return {'inconsistent': False}
        
        # This is a simplified check - in practice, would compare actual positions
        # For now, check if text coverage varies significantly
        first_coverage = first_page['text_coverage']
        
        for page_analysis in page_analyses[1:]:
            coverage = page_analysis['text_coverage']
            if abs(coverage - first_coverage) > 20:  # 20% difference
                inconsistencies.append({
                    'page': page_analysis['page_num'],
                    'coverage_diff': abs(coverage - first_coverage)
                })
        
        return {
            'inconsistent': len(inconsistencies) > 0,
            'inconsistencies': inconsistencies
        }
    
    def _check_text_density(self, page_analyses):
        """Check text density across pages."""
        densities = [p['text_coverage'] for p in page_analyses]
        
        if not densities:
            return {'average': 0, 'variance': 0}
        
        average = sum(densities) / len(densities)
        variance = sum((d - average) ** 2 for d in densities) / len(densities)
        
        return {
            'average': average,
            'variance': variance,
            'densities': densities
        }
