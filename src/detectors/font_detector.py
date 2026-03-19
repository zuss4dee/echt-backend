"""
Font Detector Module

Detects font inconsistencies that indicate text manipulation.
"""

import fitz  # PyMuPDF
from collections import Counter


class FontDetector:
    """Detects fraud indicators through font analysis."""
    
    def analyze(self, pdf_doc):
        """
        Analyze fonts for fraud indicators.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            dict: {
                'flags': list of str,
                'findings': dict with detailed findings,
                'risk_score': int (0-100),
                'font_issues': list of detected issues
            }
        """
        flags = []
        findings = {}
        risk_score = 0
        font_issues = []
        
        try:
            # Collect fonts from all pages
            all_fonts = []
            fonts_by_page = []
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_fonts = page.get_fonts()
                fonts_by_page.append(page_fonts)
                all_fonts.extend(page_fonts)
            
            if not all_fonts:
                flags.append('No Fonts Detected - Document May Be Image-Based')
                risk_score += 15
                font_issues.append('no_fonts')
                findings['no_fonts'] = True
                return {
                    'flags': flags,
                    'findings': findings,
                    'risk_score': min(risk_score, 100),
                    'font_issues': font_issues
                }
            
            # Check font embedding
            embedding_check = self._check_font_embedding(all_fonts)
            if embedding_check['issues']:
                flags.append('Font Embedding Issues Detected')
                risk_score += 20
                font_issues.append('embedding_issues')
                findings['embedding'] = embedding_check
            
            # Check font consistency
            consistency_check = self._check_font_consistency(fonts_by_page)
            if consistency_check['inconsistent']:
                flags.append('Font Inconsistency Detected')
                risk_score += 15
                font_issues.append('inconsistency')
                findings['consistency'] = consistency_check
            
            # Check for system fonts
            system_font_check = self._check_system_fonts(all_fonts)
            if system_font_check['has_system_fonts']:
                flags.append('System Fonts Detected (Potential Editing)')
                risk_score += 25
                font_issues.append('system_fonts')
                findings['system_fonts'] = system_font_check
            
            # Check font variety
            variety_check = self._check_font_variety(all_fonts)
            findings['variety'] = variety_check
            
        except Exception as e:
            flags.append(f'Font Analysis Error: {str(e)}')
            findings['error'] = str(e)
        
        return {
            'flags': flags,
            'findings': findings,
            'risk_score': min(risk_score, 100),
            'font_issues': font_issues
        }
    
    def _check_font_embedding(self, fonts):
        """Check if fonts are properly embedded."""
        issues = []
        embedded_count = 0
        not_embedded_count = 0
        
        for font in fonts:
            font_name = font.get('name', 'Unknown')
            # Check if font is embedded (ext field indicates embedding)
            ext = font.get('ext', '')
            
            # In PyMuPDF, embedded fonts typically have specific characteristics
            # Check font file reference
            if 'file' in font:
                embedded_count += 1
            else:
                not_embedded_count += 1
                if font_name not in [i.get('name') for i in issues]:
                    issues.append({
                        'name': font_name,
                        'type': font.get('type', 'Unknown'),
                        'issue': 'Font not embedded'
                    })
        
        return {
            'issues': issues,
            'embedded_count': embedded_count,
            'not_embedded_count': not_embedded_count,
            'total_fonts': len(fonts),
            'embedding_rate': embedded_count / len(fonts) if fonts else 0
        }
    
    def _check_font_consistency(self, fonts_by_page):
        """Check font consistency across pages."""
        if len(fonts_by_page) < 2:
            return {'inconsistent': False, 'reason': 'Single page document'}
        
        # Get unique font names per page
        page_font_sets = []
        for page_fonts in fonts_by_page:
            font_names = set(f.get('name', 'Unknown') for f in page_fonts)
            page_font_sets.append(font_names)
        
        # Check if fonts are consistent across pages
        base_fonts = page_font_sets[0]
        inconsistent = False
        inconsistencies = []
        
        for i, page_fonts in enumerate(page_font_sets[1:], 1):
            # Check for fonts that appear suddenly
            new_fonts = page_fonts - base_fonts
            if new_fonts:
                inconsistent = True
                inconsistencies.append({
                    'page': i + 1,
                    'new_fonts': list(new_fonts),
                    'type': 'new_fonts_appeared'
                })
            
            # Check for fonts that disappeared
            missing_fonts = base_fonts - page_fonts
            if missing_fonts and i < len(page_font_sets) - 1:  # Not last page
                inconsistent = True
                inconsistencies.append({
                    'page': i + 1,
                    'missing_fonts': list(missing_fonts),
                    'type': 'fonts_disappeared'
                })
        
        return {
            'inconsistent': inconsistent,
            'inconsistencies': inconsistencies,
            'total_pages': len(fonts_by_page)
        }
    
    def _check_system_fonts(self, fonts):
        """Check for system fonts that indicate potential editing."""
        system_font_patterns = [
            'arial', 'times', 'courier', 'helvetica', 'calibri',
            'verdana', 'tahoma', 'georgia', 'comic sans', 'impact'
        ]
        
        system_fonts_found = []
        for font in fonts:
            font_name = (font.get('name', '') or '').lower()
            for pattern in system_font_patterns:
                if pattern in font_name:
                    system_fonts_found.append(font.get('name', 'Unknown'))
                    break
        
        return {
            'has_system_fonts': len(system_fonts_found) > 0,
            'system_fonts': list(set(system_fonts_found)),
            'count': len(set(system_fonts_found))
        }
    
    def _check_font_variety(self, fonts):
        """Check font variety - too many fonts may indicate editing."""
        font_names = [f.get('name', 'Unknown') for f in fonts]
        unique_fonts = len(set(font_names))
        font_counts = Counter(font_names)
        
        # Too many different fonts may indicate editing
        suspicious = False
        if unique_fonts > 10:
            suspicious = True
        
        return {
            'unique_fonts': unique_fonts,
            'total_font_instances': len(fonts),
            'suspicious': suspicious,
            'most_common': font_counts.most_common(5)
        }
