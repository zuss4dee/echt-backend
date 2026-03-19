"""
Embedded Object Detector Module

Analyzes embedded objects for manipulation signs.
"""

import fitz  # PyMuPDF


class EmbeddedObjectDetector:
    """Detects fraud indicators through embedded object analysis."""
    
    def analyze(self, pdf_doc):
        """
        Analyze embedded objects for fraud indicators.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            dict: {
                'flags': list of str,
                'findings': dict with detailed findings,
                'risk_score': int (0-100),
                'object_issues': list of detected issues
            }
        """
        flags = []
        findings = {}
        risk_score = 0
        object_issues = []
        
        try:
            # Analyze images across all pages
            all_images = []
            image_analysis = []
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                images = page.get_images()
                
                for img_index, img in enumerate(images):
                    img_info = self._analyze_image(img, page_num, img_index)
                    all_images.append(img_info)
                    image_analysis.append(img_info)
            
            findings['total_images'] = len(all_images)
            findings['image_analysis'] = image_analysis
            
            # Check image compression consistency
            compression_check = self._check_compression_consistency(all_images)
            if compression_check['inconsistent']:
                flags.append('Inconsistent Image Compression Detected')
                risk_score += 20
                object_issues.append('compression_inconsistency')
                findings['compression'] = compression_check
            
            # Check image embedding method
            embedding_check = self._check_embedding_method(all_images)
            findings['embedding'] = embedding_check
            
            # Check for unusual image count
            if len(all_images) > 50:  # Heuristic threshold
                flags.append('Unusually High Image Count')
                risk_score += 10
                object_issues.append('high_image_count')
                findings['image_count_anomaly'] = True
            
        except Exception as e:
            flags.append(f'Embedded Object Analysis Error: {str(e)}')
            findings['error'] = str(e)
        
        return {
            'flags': flags,
            'findings': findings,
            'risk_score': min(risk_score, 100),
            'object_issues': object_issues
        }
    
    def _analyze_image(self, img, page_num, img_index):
        """Analyze a single embedded image."""
        try:
            xref = img[0]  # XREF number
            # Get image info
            base_image = img
            
            return {
                'page': page_num + 1,
                'index': img_index,
                'xref': xref,
                'width': img[2] if len(img) > 2 else 0,
                'height': img[3] if len(img) > 3 else 0,
                'colorspace': img[4] if len(img) > 4 else 'Unknown',
                'bpc': img[5] if len(img) > 5 else 0,  # bits per component
                'filter': img[7] if len(img) > 7 else 'Unknown'
            }
        except Exception as e:
            return {
                'page': page_num + 1,
                'index': img_index,
                'error': str(e)
            }
    
    def _check_compression_consistency(self, images):
        """Check if image compression is consistent."""
        if len(images) < 2:
            return {'inconsistent': False}
        
        # Get compression filters
        filters = [img.get('filter', 'Unknown') for img in images if 'error' not in img]
        
        if not filters:
            return {'inconsistent': False}
        
        # Check if all images use same compression
        unique_filters = set(filters)
        inconsistent = len(unique_filters) > 2  # Allow 2 different types
        
        return {
            'inconsistent': inconsistent,
            'unique_filters': list(unique_filters),
            'filter_distribution': {f: filters.count(f) for f in unique_filters}
        }
    
    def _check_embedding_method(self, images):
        """Check how images are embedded."""
        if not images:
            return {'method': 'none', 'count': 0}
        
        # Analyze embedding characteristics
        inline_count = 0
        object_count = 0
        
        for img in images:
            if 'xref' in img:
                object_count += 1
            else:
                inline_count += 1
        
        return {
            'method': 'mixed' if inline_count > 0 and object_count > 0 else ('inline' if inline_count > 0 else 'object'),
            'inline_count': inline_count,
            'object_count': object_count,
            'total': len(images)
        }
