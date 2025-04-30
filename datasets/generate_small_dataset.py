#!/usr/bin/env python3
"""
Generate a Small Dataset with Vocabulary for Generative Laminet Model

This script creates a compact dataset (1,000 samples) with vocabulary information
that can be trained in 1-2 hours using optimized training methods.
"""

import json
import random
from typing import Dict, List, Tuple
import re
from collections import Counter
import os

# Simple word tokenizer function to avoid nltk dependency
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and removes punctuation"""
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace and filter out empty strings
    return [token for token in text.split() if token]

# Define the semantic spaces and concepts (reduced set for faster training)
SPACES = ['temperature', 'emotion', 'complexity', 'abstraction']
CONCEPTS = {
    'temperature': ['freezing', 'cold', 'cool', 'mild', 'warm', 'hot', 'burning'],
    'emotion': ['despair', 'sad', 'melancholy', 'neutral', 'content', 'happy', 'ecstatic'],
    'complexity': ['atomic', 'simple', 'basic', 'moderate', 'complex', 'intricate', 'labyrinthine'],
    'abstraction': ['concrete', 'tangible', 'practical', 'balanced', 'theoretical', 'abstract', 'philosophical']
}

# Simplified transition patterns
PATTERNS = [
    'linear_progression',  # Move to next concept in same space
    'dimensional_shift',   # Move to different space
    'reverse'              # Move backward in sequence
]

# Define pattern weights (probability of each pattern)
PATTERN_WEIGHTS = {
    'linear_progression': 0.6,
    'dimensional_shift': 0.3,
    'reverse': 0.1
}

# Text templates for each concept (reduced for faster training)
TEXT_TEMPLATES = {
    'temperature': {
        'freezing': [
            "The arctic wind created freezing conditions that made exposed skin painful within seconds.",
            "The temperature dropped to freezing overnight, covering everything in a layer of frost.",
            "The freezing water of the mountain lake numbed his limbs almost immediately."
        ],
        'cold': [
            "The cold winter morning required several layers of clothing to stay comfortable outside.",
            "Their breath formed visible clouds in the cold air of the unheated warehouse.",
            "She shivered in the cold room, wishing she had brought a sweater."
        ],
        'cool': [
            "The evening turned cool as the sun disappeared behind the mountains.",
            "A cool breeze rustled the leaves as they sat on the porch.",
            "The cool autumn day was perfect for a long walk through the park."
        ],
        'mild': [
            "The mild weather made it comfortable to be outside in just a light jacket.",
            "Spring brought mild temperatures that were neither too hot nor too cold.",
            "The mild summer evening was perfect for dining outside."
        ],
        'warm': [
            "The warm sunlight streamed through the window, creating pools of light on the floor.",
            "They gathered around the warm fireplace, sharing stories late into the night.",
            "Her hands felt warm wrapped around the mug of freshly brewed coffee."
        ],
        'hot': [
            "The desert sun made the air hot and dry, shimmering above the sand.",
            "They sought shade to escape the hot temperatures of the summer afternoon.",
            "The soup was still hot, steam rising from the bowl."
        ],
        'burning': [
            "The metal was burning to the touch after sitting in direct sunlight all day.",
            "Her skin was burning after too many hours at the beach without sunscreen.",
            "The building was fully engulfed, flames burning bright against the night sky."
        ]
    },
    'emotion': {
        'despair': [
            "After receiving the terminal diagnosis, he sank into despair, unable to see any future.",
            "The destruction of her hometown filled her with despair for what was lost forever.",
            "Despair overwhelmed the refugees as they faced another winter without adequate shelter."
        ],
        'sad': [
            "The movie's ending left everyone in the theater feeling sad and reflective.",
            "He felt sad looking at the photographs of friends who had drifted away over the years.",
            "The abandoned puppy's eyes looked sad as it huddled in the corner of the shelter cage."
        ],
        'melancholy': [
            "The melancholy music of the violin solo brought tears to several audience members.",
            "Autumn's falling leaves always gave him a melancholy feeling about the passing of time.",
            "The empty playground had a melancholy atmosphere in the fading evening light."
        ],
        'neutral': [
            "The scientist maintained a neutral expression while presenting the controversial findings.",
            "She kept her voice neutral when discussing the sensitive political topic.",
            "His writing took a neutral stance, presenting facts without obvious bias."
        ],
        'content': [
            "After the hard climb, they sat content at the summit, enjoying the view.",
            "The cat looked content as it curled up on the sunny windowsill.",
            "The elderly couple seemed content sitting together, not needing to speak."
        ],
        'happy': [
            "The children were happy, laughing as they chased each other through the sprinklers.",
            "She was happy to receive the unexpected letter from her old friend.",
            "The dog's wagging tail showed how happy it was to see its owner return."
        ],
        'ecstatic': [
            "The team was ecstatic when they won the championship in the final seconds of the game.",
            "She was ecstatic to be accepted into her dream university program.",
            "After searching for years, the collector was ecstatic to finally find the rare edition."
        ]
    },
    'complexity': {
        'atomic': [
            "The research focused on atomic interactions that occur at scales impossible to observe directly.",
            "The atomic structure of the new material gave it unexpected properties.",
            "Understanding atomic theory was essential for students continuing to advanced physics."
        ],
        'simple': [
            "The recipe was simple, requiring only five common ingredients.",
            "She preferred simple explanations over convoluted theories.",
            "The simple pleasure of a walk in nature restored his peace of mind."
        ],
        'basic': [
            "The course covered basic concepts that would form the foundation for advanced studies.",
            "Their shelter provided basic protection from the elements, but little comfort.",
            "The app included basic functionality but lacked the features of premium alternatives."
        ],
        'moderate': [
            "The hike was of moderate difficulty, challenging but accessible to most people.",
            "The exam tested moderate knowledge of the subject, neither too easy nor too difficult.",
            "The project was of moderate complexity, requiring a team but not specialists."
        ],
        'complex': [
            "The software had a complex architecture that made modifications risky.",
            "The novel presented a complex narrative with multiple timelines and perspectives.",
            "The immune system is a complex network of cells and signals working in concert."
        ],
        'intricate': [
            "The watch contained intricate gears and mechanisms assembled by skilled craftsmen.",
            "She created intricate patterns of lace that took months to complete.",
            "The spy novel had an intricate plot with unexpected connections between characters."
        ],
        'labyrinthine': [
            "The ancient city had labyrinthine streets that confused even longtime residents.",
            "The legal document was labyrinthine, with clauses that referenced each other recursively.",
            "The cave system was labyrinthine, requiring experienced guides for exploration."
        ]
    },
    'abstraction': {
        'concrete': [
            "She preferred concrete examples rather than abstract theories to explain the concept.",
            "The company wanted concrete results that could be measured and quantified.",
            "The witness provided concrete evidence that contradicted the defendant's story."
        ],
        'tangible': [
            "The project delivered tangible benefits to the community within the first year.",
            "He wanted something tangible to remember his journey, not just digital photos.",
            "The museum allowed visitors to have tangible experiences with historical artifacts."
        ],
        'practical': [
            "Her approach was practical, focusing on solutions rather than assigning blame.",
            "The course emphasized practical skills that students could apply immediately.",
            "He offered practical advice based on years of experience in the field."
        ],
        'balanced': [
            "The news program presented a balanced perspective on the controversial issue.",
            "She took a balanced approach to parenting, providing both structure and freedom.",
            "The chef created a balanced menu with flavors that complemented each other."
        ],
        'theoretical': [
            "The solution was theoretical and had not yet been tested in real-world conditions.",
            "His work remained largely theoretical, based on mathematical models rather than experiments.",
            "The theoretical physics paper proposed ideas that might not be testable for decades."
        ],
        'abstract': [
            "The painting used abstract shapes and colors to evoke emotional responses.",
            "The philosopher spoke in abstract terms about concepts of justice and freedom.",
            "She found it difficult to grasp such abstract concepts without concrete examples."
        ],
        'philosophical': [
            "The novel raised philosophical questions about the nature of consciousness.",
            "They engaged in philosophical debate long into the night, exploring fundamental questions.",
            "The documentary examined the philosophical implications of new technologies."
        ]
    }
}

def get_valid_transitions(source_space: str, source_concept: str, pattern: str) -> List[Tuple[str, str]]:
    """
    Get valid target transitions based on source and pattern.
    Returns a list of (target_space, target_concept) tuples.
    """
    valid_targets = []
    
    # Get index of source concept in its space
    if source_space not in CONCEPTS:
        return []
    
    source_concepts = CONCEPTS[source_space]
    if source_concept not in source_concepts:
        return []
    
    source_idx = source_concepts.index(source_concept)
    
    if pattern == 'linear_progression':
        # Move to next concept in same space
        if source_idx < len(source_concepts) - 1:
            valid_targets.append((source_space, source_concepts[source_idx + 1]))
    
    elif pattern == 'dimensional_shift':
        # Move to a concept in another space
        for other_space in SPACES:
            if other_space != source_space:
                # Try to find similar position in other space
                other_concepts = CONCEPTS[other_space]
                relative_pos = min(source_idx, len(other_concepts) - 1)
                valid_targets.append((other_space, other_concepts[relative_pos]))
    
    elif pattern == 'reverse':
        # Move backward in the sequence
        if source_idx > 0:
            valid_targets.append((source_space, source_concepts[source_idx - 1]))
    
    return valid_targets

def generate_sample(sample_id: int) -> Dict:
    """Generate a single sample with metadata and text"""
    # Select transition pattern
    pattern = random.choices(
        population=PATTERNS,
        weights=[PATTERN_WEIGHTS[p] for p in PATTERNS],
        k=1
    )[0]
    
    # Select source concept and space
    source_space = random.choice(SPACES)
    source_concept = random.choice(CONCEPTS[source_space])
    
    # Get valid transitions for this pattern
    valid_transitions = get_valid_transitions(source_space, source_concept, pattern)
    
    # If no valid transitions, try a different pattern
    if not valid_transitions:
        # Try linear progression
        pattern = 'linear_progression'
        valid_transitions = get_valid_transitions(source_space, source_concept, pattern)
        
        # If still no valid transitions, use dimensional shift
        if not valid_transitions:
            pattern = 'dimensional_shift'
            valid_transitions = get_valid_transitions(source_space, source_concept, pattern)
            
            # Last resort - pick any valid target
            if not valid_transitions:
                target_space = random.choice(SPACES)
                target_concept = random.choice(CONCEPTS[target_space])
            else:
                target_space, target_concept = random.choice(valid_transitions)
        else:
            target_space, target_concept = random.choice(valid_transitions)
    else:
        target_space, target_concept = random.choice(valid_transitions)
    
    # Get text templates for source and target
    source_text = random.choice(TEXT_TEMPLATES[source_space][source_concept])
    target_text = random.choice(TEXT_TEMPLATES[target_space][target_concept])
    
    # Create the sample
    sample = {
        "sample_id": sample_id,
        "source_space": source_space,
        "source_concept": source_concept,
        "target_space": target_space,
        "target_concept": target_concept,
        "transition_pattern": pattern,
        "source_text": source_text,
        "target_text": target_text
    }
    
    return sample

def build_vocabulary(samples):
    """
    Build a vocabulary from all the text in the samples.
    Returns a dictionary mapping words to indices and special tokens.
    """
    # Collect all text from samples
    all_text = []
    for sample in samples:
        all_text.append(sample['source_text'])
        all_text.append(sample['target_text'])
    
    # Combine all text and tokenize using our simple tokenizer
    combined_text = " ".join(all_text)
    tokens = simple_tokenize(combined_text)
    
    # Count word frequencies
    word_counts = Counter(tokens)
    
    # Keep only words that appear at least twice
    frequent_words = [word for word, count in word_counts.items() if count >= 2]
    
    # Add special tokens
    special_tokens = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    
    # Create word-to-index mapping
    word_to_idx = {word: idx+len(special_tokens) for idx, word in enumerate(frequent_words)}
    
    # Add special tokens to vocabulary
    for token, idx in special_tokens.items():
        word_to_idx[token] = idx
    
    # Create index-to-word mapping
    idx_to_word = {str(idx): word for word, idx in word_to_idx.items()}
    
    return {
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "special_tokens": special_tokens,
        "vocab_size": len(word_to_idx)
    }

def main():
    """Generate and save 1,000 samples with vocabulary"""
    num_samples = 1000
    samples = []
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    for i in range(num_samples):
        sample = generate_sample(i)
        samples.append(sample)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} samples")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocabulary = build_vocabulary(samples)
    
    # Save samples to file
    output_dir = os.path.dirname(os.path.abspath(__file__))
    samples_file = os.path.join(output_dir, "laminet_samples_1k.json")
    vocab_file = os.path.join(output_dir, "laminet_vocabulary.json")
    
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    with open(vocab_file, 'w') as f:
        json.dump(vocabulary, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {samples_file}")
    print(f"Saved vocabulary with {vocabulary['vocab_size']} words to {vocab_file}")
    
    # Print some statistics
    pattern_counts = {}
    space_counts = {"source": {}, "target": {}}
    
    for sample in samples:
        # Count patterns
        pattern = sample["transition_pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Count spaces
        source_space = sample["source_space"]
        target_space = sample["target_space"]
        space_counts["source"][source_space] = space_counts["source"].get(source_space, 0) + 1
        space_counts["target"][target_space] = space_counts["target"].get(target_space, 0) + 1
    
    print("\nPattern distribution:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} samples ({count/num_samples*100:.1f}%)")
    
    print("\nSource space distribution:")
    for space, count in space_counts["source"].items():
        print(f"  {space}: {count} samples ({count/num_samples*100:.1f}%)")
    
    print("\nTarget space distribution:")
    for space, count in space_counts["target"].items():
        print(f"  {space}: {count} samples ({count/num_samples*100:.1f}%)")
    
    print(f"\nVocabulary size: {vocabulary['vocab_size']} words")

if __name__ == "__main__":
    main() 