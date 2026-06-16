#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""
Generate AprilTag images and printable PDFs specifically for the tagStandard41h12 family.
Optimized for the pick-and-place project standards.
"""

import argparse
import os
import cv2
import numpy as np
from fpdf import FPDF

# Paper sizes in mm
PAPER_SIZES = {
    "A4": (210, 297),
    "Letter": (215.9, 279.4),
}

# Robotics presets: (tag_size_mm, count, starting_id)
PRESETS = {
    "workspace_frame": (40.0, 4, 12),  # IDs 12-15 are standard for workspace_frame
    "cube": (20.0, 6, 0),        # IDs 0-5 are standard for cubes
    "drop_box": (60.0, 4, 8),    # IDs 8-11 are standard for drop boxes
}

# Bit patterns for tagStandard41h12 (IDs 0-25) extracted from official AprilTag 3.
# 1 = Black, 0 = White. 9x9 grid.
TAG_41H12_BITS = {
    0: [[0, 0, 1, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0, 1, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1, 1]],
    1: [[0, 0, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 0, 0, 1, 1, 0]],
    2: [[0, 0, 1, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1, 1, 1]],
    3: [[0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 1, 1, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1, 1, 1, 0]],
    4: [[0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 1, 1, 1, 1]],
    5: [[0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 1]],
    6: [[0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0]],
    7: [[0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1]],
    8: [[0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0, 0]],
    9: [[0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1, 0, 1]],
    10: [[0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0, 1, 0, 0]],
    11: [[0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1, 0, 1]],
    12: [[0, 0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1, 0, 0]],
    13: [[0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 0, 0, 1, 0]],
    14: [[0, 0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1]],
    15: [[0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 0, 1, 0]],
    16: [[0, 0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0, 1, 1]],
    17: [[0, 0, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 1, 0]],
    18: [[0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1]],
    19: [[0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 0, 1]],
    20: [[0, 0, 0, 1, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 0, 1]],
    21: [[0, 0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 1, 0, 0, 1, 0, 1]],
    22: [[0, 0, 0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1, 0, 0]],
    23: [[0, 0, 0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1, 0, 1]],
    24: [[0, 0, 0, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 1, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 1, 1, 0, 0]],
    25: [[0, 0, 0, 1, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0, 0, 1, 0]],
}

def get_tag_img(tag_id, size_px):
    if tag_id not in TAG_41H12_BITS:
        print(f"Warning: ID {tag_id} not in hardcoded 41h12 list. Using empty tag.")
        return np.ones((size_px, size_px), dtype=np.uint8) * 255
    
    bits = np.array(TAG_41H12_BITS[tag_id], dtype=np.uint8)
    cell_size = size_px // 9
    img = np.ones((size_px, size_px), dtype=np.uint8) * 255
    for r in range(9):
        for c in range(9):
            if bits[r, c] == 1:
                img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = 0
    return img

def generate_tags(tag_ids, output_dir, tag_size_mm, margin_mm, paper_format):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {len(tag_ids)} tags (tagStandard41h12)...")
    
    tag_data = []
    for tag_id in tag_ids:
        # Use a high enough resolution for the raw PNGs (multiple of 9)
        size_px = 450
        tag_img = get_tag_img(tag_id, size_px)
            
        raw_path = os.path.join(output_dir, f"raw_41h12_{tag_id:05d}.png")
        cv2.imwrite(raw_path, tag_img)
        
        # Labeled version for preview
        margin_px = int(size_px * 0.1)
        label_h_px = int(size_px * 0.15)
        final_size_px = size_px + 2 * margin_px
        canvas = np.ones((final_size_px + label_h_px, final_size_px), dtype=np.uint8) * 255
        canvas[margin_px:margin_px+size_px, margin_px:margin_px+size_px] = tag_img
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = size_px / 400.0
        label_text = f"ID: {tag_id} (41h12)"
        text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
        cv2.putText(canvas, label_text, ((final_size_px - text_size[0]) // 2, final_size_px + 10), 
                    font, font_scale, (0,), 1, cv2.LINE_AA)
        
        fancy_path = os.path.join(output_dir, f"41h12_{tag_id:05d}.png")
        cv2.imwrite(fancy_path, canvas)
        
        tag_data.append({"id": tag_id, "raw": raw_path, "fancy": fancy_path})

    create_pdf(tag_data, tag_size_mm, margin_mm, output_dir, paper_format)

def create_pdf(tag_data, tag_size_mm, gap_mm, output_dir, paper_format):
    paper_w, paper_h = PAPER_SIZES.get(paper_format, PAPER_SIZES["A4"])
    page_margin = 15
    pdf = FPDF(orientation='P', unit='mm', format=paper_format)
    pdf.set_auto_page_break(False)
    
    item_w = tag_size_mm
    item_h = tag_size_mm + 5
    usable_w = paper_w - 2 * page_margin
    usable_h = paper_h - 2 * page_margin - 10 
    
    cols = max(1, int((usable_w + gap_mm) // (item_w + gap_mm)))
    rows = max(1, int((usable_h + gap_mm) // (item_h + gap_mm)))
    tags_per_page = cols * rows
    
    for i, data in enumerate(tag_data):
        item_idx = i % tags_per_page
        if item_idx == 0:
            pdf.add_page()
            pdf.set_font("Helvetica", style='B', size=10)
            pdf.text(page_margin, 12, f"AprilTag Standard 41h12 | Target Size: {tag_size_mm}mm")
            pdf.set_font("Helvetica", size=8)
            pdf.text(page_margin, 16, f"Print scale must be 100%. Black border: {tag_size_mm}mm.")
            
        col = item_idx % cols
        row = item_idx // cols
        x = page_margin + col * (item_w + gap_mm)
        y = page_margin + 15 + row * (item_h + gap_mm)
        
        pdf.image(data["raw"], x=x, y=y, w=tag_size_mm)
        pdf.set_font("Helvetica", size=7)
        pdf.text(x, y + tag_size_mm + 3, f"ID: {data['id']}")

    filename = f"apriltags_41h12_{int(tag_size_mm)}mm_{paper_format}.pdf"
    pdf_path = os.path.join(output_dir, filename)
    pdf.output(pdf_path)
    print(f"PDF generated: {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate AprilTag 41h12 PDFs.")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                        help="Robotics presets: workspace_frame (40mm, IDs 12-15), cube (20mm, IDs 0-5), drop_box (60mm, IDs 8-11)")

    parser.add_argument("--ids", type=int, nargs='+', help="Specific IDs to generate (0-25)")
    parser.add_argument("--tag_size_mm", type=float, default=40.0, help="Manual tag size in mm")
    parser.add_argument("--output_dir", type=str, default="out/apriltags", help="Output directory")
    parser.add_argument("--paper", type=str, default="A4", choices=PAPER_SIZES.keys(), help="Paper format")
    
    args = parser.parse_args()
    
    tag_size = args.tag_size_mm
    
    if args.preset:
        tag_size, count, start_id = PRESETS[args.preset]
        tag_ids = list(range(start_id, start_id + count))
        print(f"Using preset '{args.preset}': {tag_size}mm tags, IDs {tag_ids}")
    elif args.ids:
        tag_ids = args.ids
    else:
        # Default to 40mm calibration tags
        tag_ids = [12, 13, 14, 15]
        print("No IDs or preset specified. Defaulting to workspace_frame tags (12-15).")

    generate_tags(tag_ids, args.output_dir, tag_size, 10.0, args.paper)

if __name__ == "__main__":
    main()
