"""
Layout Detector Module

Checks for text alignment and font anomalies using layout analysis.
"""

import fitz  # PyMuPDF
from collections import Counter


class LayoutDetector:
    """Detects fraud indicators through layout analysis."""
    
    def analyze(self, pdf_doc):
        """
        Analyze layout for fraud indicators.
        
        Args:
            pdf_doc: PyMuPDF document object
            
        Returns:
            dict: {
                'flags': list of str,
                'findings': dict with detailed findings,
                'risk_score': int (0-100),
                'layout_issues': list of detected issues
            }
        """
        flags = []
        findings = {}
        risk_score = 0
        layout_issues = []
        
        try:
            # Analyze each page
            page_layouts = []
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_layout = self._analyze_page_layout(page, page_num)
                page_layouts.append(page_layout)
            
            findings['page_layouts'] = page_layouts
            
            # Check text alignment consistency
            alignment_check = self._check_alignment_consistency(page_layouts)
            if alignment_check['inconsistent']:
                flags.append('Text Alignment Inconsistencies Detected')
                risk_score += 20
                layout_issues.append('alignment_issues')
                findings['alignment'] = alignment_check
            
            # Check font size consistency
            font_size_check = self._check_font_size_consistency(page_layouts)
            if font_size_check['inconsistent']:
                flags.append('Font Size Inconsistencies Detected')
                risk_score += 15
                layout_issues.append('font_size_issues')
                findings['font_sizes'] = font_size_check
            
            # Check margin consistency
            margin_check = self._check_margin_consistency(page_layouts)
            if margin_check['inconsistent']:
                flags.append('Margin Inconsistencies Detected')
                risk_score += 15
                layout_issues.append('margin_issues')
                findings['margins'] = margin_check
            
            # Check line spacing
            spacing_check = self._check_line_spacing(page_layouts)
            if spacing_check['inconsistent']:
                flags.append('Line Spacing Inconsistencies')
                risk_score += 10
                layout_issues.append('spacing_issues')
                findings['line_spacing'] = spacing_check
            
            # Check header/footer consistency
            header_footer_check = self._check_header_footer(page_layouts)
            findings['header_footer'] = header_footer_check
            
        except Exception as e:
            flags.append(f'Layout Analysis Error: {str(e)}')
            findings['error'] = str(e)
        
        return {
            'flags': flags,
            'findings': findings,
            'risk_score': min(risk_score, 100),
            'layout_issues': layout_issues
        }
    
    def _analyze_page_layout(self, page, page_num):
        """Analyze layout of a single page."""
        # Get text blocks with positions
        text_blocks = page.get_text("blocks")
        
        # Get text as dict for detailed analysis
        text_dict = page.get_text("dict")
        
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Analyze text block positions
        left_margins = []
        right_margins = []
        top_margins = []
        bottom_margins = []
        font_sizes = []
        line_heights = []
        
        for block in text_blocks:
            if len(block) >= 5:
                x0, y0, x1, y1 = block[:4]
                
                # Calculate margins
                left_margins.append(x0)
                right_margins.append(page_width - x1)
                top_margins.append(y0)
                bottom_margins.append(page_height - y1)
        
        # Extract font sizes from text dict
        for block in text_dict.get('blocks', []):
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    font_size = span.get('size', 0)
                    if font_size > 0:
                        font_sizes.append(font_size)
                    
                    # Calculate line height
                    bbox = span.get('bbox', [])
                    if len(bbox) >= 4:
                        line_height = bbox[3] - bbox[1]
                        if line_height > 0:
                            line_heights.append(line_height)
        
        return {
            'page_num': page_num + 1,
            'page_width': page_width,
            'page_height': page_height,
            'text_block_count': len(text_blocks),
            'left_margins': left_margins,
            'right_margins': right_margins,
            'top_margins': top_margins,
            'bottom_margins': bottom_margins,
            'font_sizes': font_sizes,
            'line_heights': line_heights,
            'avg_left_margin': sum(left_margins) / len(left_margins) if left_margins else 0,
            'avg_right_margin': sum(right_margins) / len(right_margins) if right_margins else 0,
            'avg_top_margin': sum(top_margins) / len(top_margins) if top_margins else 0,
            'avg_bottom_margin': sum(bottom_margins) / len(bottom_margins) if bottom_margins else 0
        }
    
    def _check_alignment_consistency(self, page_layouts):
        """Check if text alignment is consistent across pages."""
        if len(page_layouts) < 2:
            return {'inconsistent': False}
        
        # Compare left margins (most important for alignment)
        first_page = page_layouts[0]
        base_left_margin = first_page['avg_left_margin']
        
        inconsistencies = []
        for page_layout in page_layouts[1:]:
            left_margin = page_layout['avg_left_margin']
            if base_left_margin > 0:
                diff_percent = abs(left_margin - base_left_margin) / base_left_margin * 100
                if diff_percent > 10:  # More than 10% difference
                    inconsistencies.append({
                        'page': page_layout['page_num'],
                        'margin_diff': diff_percent
                    })
        
        return {
            'inconsistent': len(inconsistencies) > 0,
            'inconsistencies': inconsistencies,
            'base_margin': base_left_margin
        }
    
    def _check_font_size_consistency(self, page_layouts):
        """Check if font sizes are consistent."""
        if len(page_layouts) < 2:
            return {'inconsistent': False}
        
        # Get average font size per page
        page_avg_sizes = []
        for page_layout in page_layouts:
            font_sizes = page_layout['font_sizes']
            if font_sizes:
                avg_size = sum(font_sizes) / len(font_sizes)
                page_avg_sizes.append(avg_size)
            else:
                page_avg_sizes.append(0)
        
        if not page_avg_sizes or all(s == 0 for s in page_avg_sizes):
            return {'inconsistent': False}
        
        # Check for significant variations
        base_size = page_avg_sizes[0]
        inconsistencies = []
        
        for i, size in enumerate(page_avg_sizes[1:], 1):
            if base_size > 0:
                diff_percent = abs(size - base_size) / base_size * 100
                if diff_percent > 20:  # More than 20% difference
                    inconsistencies.append({
                        'page': i + 1,
                        'size_diff': diff_percent,
                        'size': size
                    })
        
        return {
            'inconsistent': len(inconsistencies) > 0,
            'inconsistencies': inconsistencies,
            'avg_sizes': page_avg_sizes
        }
    
    def _check_margin_consistency(self, page_layouts):
        """Check if margins are consistent across pages."""
        if len(page_layouts) < 2:
            return {'inconsistent': False}
        
        first_page = page_layouts[0]
        base_margins = {
            'left': first_page['avg_left_margin'],
            'right': first_page['avg_right_margin'],
            'top': first_page['avg_top_margin'],
            'bottom': first_page['avg_bottom_margin']
        }
        
        inconsistencies = []
        for page_layout in page_layouts[1:]:
            for margin_type in ['left', 'right', 'top', 'bottom']:
                base = base_margins[margin_type]
                current = page_layout[f'avg_{margin_type}_margin']
                
                if base > 0:
                    diff_percent = abs(current - base) / base * 100
                    if diff_percent > 15:  # More than 15% difference
                        inconsistencies.append({
                            'page': page_layout['page_num'],
                            'margin_type': margin_type,
                            'diff_percent': diff_percent
                        })
        
        return {
            'inconsistent': len(inconsistencies) > 0,
            'inconsistencies': inconsistencies
        }
    
    def _check_line_spacing(self, page_layouts):
        """Check line spacing consistency."""
        if len(page_layouts) < 2:
            return {'inconsistent': False}
        
        # Get average line height per page
        page_avg_heights = []
        for page_layout in page_layouts:
            line_heights = page_layout['line_heights']
            if line_heights:
                avg_height = sum(line_heights) / len(line_heights)
                page_avg_heights.append(avg_height)
            else:
                page_avg_heights.append(0)
        
        if not page_avg_heights or all(h == 0 for h in page_avg_heights):
            return {'inconsistent': False}
        
        base_height = page_avg_heights[0]
        inconsistencies = []
        
        for i, height in enumerate(page_avg_heights[1:], 1):
            if base_height > 0:
                diff_percent = abs(height - base_height) / base_height * 100
                if diff_percent > 25:  # More than 25% difference
                    inconsistencies.append({
                        'page': i + 1,
                        'height_diff': diff_percent
                    })
        
        return {
            'inconsistent': len(inconsistencies) > 0,
            'inconsistencies': inconsistencies
        }
    
    def _check_header_footer(self, page_layouts):
        """Check header/footer consistency."""
        if len(page_layouts) < 2:
            return {'consistent': True}
        
        # Check top margins (headers) and bottom margins (footers)
        top_margins = [p['avg_top_margin'] for p in page_layouts]
        bottom_margins = [p['avg_bottom_margin'] for p in page_layouts]
        
        top_consistent = max(top_margins) - min(top_margins) < 10 if top_margins else True
        bottom_consistent = max(bottom_margins) - min(bottom_margins) < 10 if bottom_margins else True
        
        return {
            'consistent': top_consistent and bottom_consistent,
            'top_margins': top_margins,
            'bottom_margins': bottom_margins
        }