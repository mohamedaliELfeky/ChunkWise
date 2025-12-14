"""
Pytest Configuration and Fixtures

Shared fixtures for ChunkWise tests.
"""

import pytest


# Sample texts for testing
ENGLISH_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, generative or creative tools, automated decision-making, and competing at the highest level in strategic game systems.

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
"""

ARABIC_TEXT = """
الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف إلى إنشاء آلات ذكية تعمل وتتفاعل مثل البشر. بعض الأنشطة التي صممت لها أجهزة الكمبيوتر ذات الذكاء الاصطناعي تشمل التعرف على الكلام والتعلم والتخطيط وحل المشكلات.

يعتبر الذكاء الاصطناعي مجالاً متعدد التخصصات، حيث يجمع بين علوم الحاسوب والرياضيات وعلم النفس واللغويات. منذ بدايته، أثار الذكاء الاصطناعي جدلاً فلسفياً حول طبيعة العقل البشري.

تطبيقات الذكاء الاصطناعي متعددة ومتنوعة، وتشمل محركات البحث المتقدمة وأنظمة التوصية والتعرف على الكلام والسيارات ذاتية القيادة والأدوات الإبداعية.
"""

MIXED_TEXT = """
Welcome to ChunkWise! مرحباً بكم في تشنك وايز

This is a demonstration of mixed Arabic and English text processing.
هذا عرض توضيحي لمعالجة النصوص العربية والإنجليزية المختلطة.

ChunkWise supports multiple languages including Arabic العربية and English الإنجليزية.
"""

MARKDOWN_TEXT = """# Introduction

This is the introduction section.

## Background

Here is some background information about the topic. It contains multiple paragraphs and covers various aspects of the subject matter.

### Technical Details

The technical implementation uses several advanced techniques:
- Technique one
- Technique two
- Technique three

## Methodology

The methodology section describes the approach taken.

### Data Collection

Data was collected from multiple sources.

### Analysis

The analysis phase involved several steps.

## Results

Here are the main results of the study.

## Conclusion

This is the conclusion of the document.
"""

CODE_TEXT = """
import numpy as np
from typing import List, Optional

class DataProcessor:
    \"\"\"Process data for analysis.\"\"\"

    def __init__(self, config: dict):
        self.config = config
        self.data = None

    def load_data(self, path: str) -> None:
        \"\"\"Load data from file.\"\"\"
        self.data = np.load(path)

    def process(self) -> np.ndarray:
        \"\"\"Process the loaded data.\"\"\"
        if self.data is None:
            raise ValueError("No data loaded")
        return np.mean(self.data, axis=0)


def helper_function(x: int, y: int) -> int:
    \"\"\"A simple helper function.\"\"\"
    return x + y


class AnotherClass:
    \"\"\"Another class for demonstration.\"\"\"

    def method_one(self):
        pass

    def method_two(self):
        pass
"""


@pytest.fixture
def english_text():
    """Provide sample English text."""
    return ENGLISH_TEXT


@pytest.fixture
def arabic_text():
    """Provide sample Arabic text."""
    return ARABIC_TEXT


@pytest.fixture
def mixed_text():
    """Provide sample mixed language text."""
    return MIXED_TEXT


@pytest.fixture
def markdown_text():
    """Provide sample Markdown text."""
    return MARKDOWN_TEXT


@pytest.fixture
def code_text():
    """Provide sample code text."""
    return CODE_TEXT


@pytest.fixture
def short_text():
    """Provide a short text for edge cases."""
    return "Hello world. This is a short test."


@pytest.fixture
def empty_text():
    """Provide empty text for edge cases."""
    return ""
