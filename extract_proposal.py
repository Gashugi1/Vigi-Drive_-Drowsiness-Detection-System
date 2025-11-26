#!/usr/bin/env python3
"""Extract text from Word document"""
from docx import Document
import sys

def extract_text(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

if __name__ == "__main__":
    text = extract_text("presentation/148907 - Proposal Document chap 1,2 & 3  (1).docx")
    print(text)
