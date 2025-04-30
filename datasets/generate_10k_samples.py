#!/usr/bin/env python3
"""
Generate 10,000 Samples for Laminet Model Training

This script generates a JSON file with 10,000 samples for training the Laminet model.
Each sample includes metadata and text for source and target concepts.
"""

import json
import random
from typing import Dict, List, Tuple

# Define the semantic spaces and concepts
SPACES = ['temperature', 'emotion', 'complexity', 'abstraction', 'composite']
CONCEPTS = {
    'temperature': ['freezing', 'cold', 'cool', 'mild', 'warm', 'hot', 'burning'],
    'emotion': ['despair', 'sad', 'melancholy', 'neutral', 'content', 'happy', 'ecstatic'],
    'complexity': ['atomic', 'simple', 'basic', 'moderate', 'complex', 'intricate', 'labyrinthine'],
    'abstraction': ['concrete', 'tangible', 'practical', 'balanced', 'theoretical', 'abstract', 'philosophical'],
    'composite': ['memory', 'creativity', 'reasoning', 'intuition']
}

# Define transition patterns
PATTERNS = [
    'linear_progression',  # Move to next concept in same space
    'concept_leap',        # Skip a concept in same space
    'dimensional_shift',   # Move to different space
    'compositional',       # Move to composite concept
    'reverse'              # Move backward in sequence
]

# Define pattern weights (probability of each pattern)
PATTERN_WEIGHTS = {
    'linear_progression': 0.4,
    'concept_leap': 0.2,
    'dimensional_shift': 0.2,
    'compositional': 0.1,
    'reverse': 0.1
}

# Text templates for each concept
TEXT_TEMPLATES = {
    'temperature': {
        'freezing': [
            "The arctic wind created freezing conditions that made exposed skin painful within seconds.",
            "The temperature dropped to freezing overnight, covering everything in a layer of frost.",
            "The freezing water of the mountain lake numbed his limbs almost immediately.",
            "Winter brought freezing temperatures that transformed the landscape into a crystalline wonderland.",
            "The freezing rain created dangerous conditions on the roads and sidewalks."
        ],
        'cold': [
            "The cold winter morning required several layers of clothing to stay comfortable outside.",
            "Their breath formed visible clouds in the cold air of the unheated warehouse.",
            "The metal railing felt cold against her hand as she descended the stairs.",
            "The cold water of the stream provided relief during the hiking trip.",
            "She shivered in the cold room, wishing she had brought a sweater."
        ],
        'cool': [
            "The evening turned cool as the sun disappeared behind the mountains.",
            "The cool basement provided a welcome escape from the summer heat.",
            "A cool breeze rustled the leaves as they sat on the porch.",
            "The stone walls kept the interior of the castle cool even during hot days.",
            "The cool autumn day was perfect for a long walk through the park."
        ],
        'mild': [
            "The mild weather made it comfortable to be outside in just a light jacket.",
            "Spring brought mild temperatures that were neither too hot nor too cold.",
            "The region's mild climate supported a diverse range of plant life.",
            "The mild winter meant they rarely needed to use the heating system.",
            "The mild summer evening was perfect for dining outside."
        ],
        'warm': [
            "The warm sunlight streamed through the window, creating pools of light on the floor.",
            "They gathered around the warm fireplace, sharing stories late into the night.",
            "The sand felt warm beneath their feet as they walked along the beach.",
            "The kitchen was warm and fragrant with the smell of baking bread.",
            "Her hands felt warm wrapped around the mug of freshly brewed coffee."
        ],
        'hot': [
            "The desert sun made the air hot and dry, shimmering above the sand.",
            "The hot pavement burned through the soles of their shoes in the midday sun.",
            "They sought shade to escape the hot temperatures of the summer afternoon.",
            "The engine felt hot after the long drive through the mountains.",
            "The soup was still hot, steam rising from the bowl."
        ],
        'burning': [
            "The metal was burning to the touch after sitting in direct sunlight all day.",
            "The wildfire created burning conditions across thousands of acres of forest.",
            "Her skin was burning after too many hours at the beach without sunscreen.",
            "The chile pepper left a burning sensation on his tongue and lips.",
            "The building was fully engulfed, flames burning bright against the night sky."
        ]
    },
    'emotion': {
        'despair': [
            "After receiving the terminal diagnosis, he sank into despair, unable to see any future.",
            "The destruction of her hometown filled her with despair for what was lost forever.",
            "His face showed utter despair when he realized all his work had been deleted.",
            "The prisoner's despair grew with each passing year without contact from the outside world.",
            "Despair overwhelmed the refugees as they faced another winter without adequate shelter."
        ],
        'sad': [
            "The movie's ending left everyone in the theater feeling sad and reflective.",
            "He felt sad looking at the photographs of friends who had drifted away over the years.",
            "The withered flowers on the grave created a sad reminder of time's passage.",
            "Her voice sounded sad as she described the changes in her old neighborhood.",
            "The abandoned puppy's eyes looked sad as it huddled in the corner of the shelter cage."
        ],
        'melancholy': [
            "The melancholy music of the violin solo brought tears to several audience members.",
            "Autumn's falling leaves always gave him a melancholy feeling about the passing of time.",
            "She felt a pleasant melancholy as she packed away her daughter's baby clothes.",
            "The empty playground had a melancholy atmosphere in the fading evening light.",
            "His paintings captured the melancholy beauty of the foggy coastline."
        ],
        'neutral': [
            "The scientist maintained a neutral expression while presenting the controversial findings.",
            "The beige walls created a neutral backdrop for the colorful artwork.",
            "She kept her voice neutral when discussing the sensitive political topic.",
            "The judge's neutral demeanor gave no indication of which way she might rule.",
            "His writing took a neutral stance, presenting facts without obvious bias."
        ],
        'content': [
            "After the hard climb, they sat content at the summit, enjoying the view.",
            "The cat looked content as it curled up on the sunny windowsill.",
            "The simple meal left them feeling content and satisfied.",
            "She felt content with her decision to take the less conventional career path.",
            "The elderly couple seemed content sitting together, not needing to speak."
        ],
        'happy': [
            "The children were happy, laughing as they chased each other through the sprinklers.",
            "She was happy to receive the unexpected letter from her old friend.",
            "The dog's wagging tail showed how happy it was to see its owner return.",
            "Their faces looked happy in the photograph from their wedding day.",
            "He felt genuinely happy for the first time in years after moving to the countryside."
        ],
        'ecstatic': [
            "The team was ecstatic when they won the championship in the final seconds of the game.",
            "She was ecstatic to be accepted into her dream university program.",
            "The audience was ecstatic, giving a standing ovation that lasted ten minutes.",
            "His parents were ecstatic when he announced he was moving back to his hometown.",
            "After searching for years, the collector was ecstatic to finally find the rare edition."
        ]
    },
    'complexity': {
        'atomic': [
            "The research focused on atomic interactions that occur at scales impossible to observe directly.",
            "The atomic structure of the new material gave it unexpected properties.",
            "The atomic clock measured time with unprecedented precision.",
            "Understanding atomic theory was essential for students continuing to advanced physics.",
            "The atomic components of the experiment had to be handled with specialized equipment."
        ],
        'simple': [
            "The recipe was simple, requiring only five common ingredients.",
            "She preferred simple explanations over convoluted theories.",
            "The cabin had a simple design with just one room and minimal furnishings.",
            "The simple pleasure of a walk in nature restored his peace of mind.",
            "He gave a simple answer to a question that could have been answered with great complexity."
        ],
        'basic': [
            "The course covered basic concepts that would form the foundation for advanced studies.",
            "Their shelter provided basic protection from the elements, but little comfort.",
            "She had a basic understanding of several languages but wasn't fluent in any.",
            "The app included basic functionality but lacked the features of premium alternatives.",
            "He completed the basic training required for all new employees."
        ],
        'moderate': [
            "The hike was of moderate difficulty, challenging but accessible to most people.",
            "The exam tested moderate knowledge of the subject, neither too easy nor too difficult.",
            "They lived in moderate comfort, neither luxurious nor sparse.",
            "The sauce had moderate spiciness that added flavor without overwhelming the dish.",
            "The project was of moderate complexity, requiring a team but not specialists."
        ],
        'complex': [
            "The software had a complex architecture that made modifications risky.",
            "The novel presented a complex narrative with multiple timelines and perspectives.",
            "Modern aircraft rely on complex systems with multiple redundancies for safety.",
            "The immune system is a complex network of cells and signals working in concert.",
            "The negotiations involved complex legal and cultural factors across multiple countries."
        ],
        'intricate': [
            "The watch contained intricate gears and mechanisms assembled by skilled craftsmen.",
            "She created intricate patterns of lace that took months to complete.",
            "The temple walls were covered with intricate carvings depicting ancient stories.",
            "The spy novel had an intricate plot with unexpected connections between characters.",
            "The ecosystem represented an intricate balance that could be disrupted by small changes."
        ],
        'labyrinthine': [
            "The ancient city had labyrinthine streets that confused even longtime residents.",
            "The legal document was labyrinthine, with clauses that referenced each other recursively.",
            "The corporation had a labyrinthine structure designed to minimize tax liability.",
            "The novel's labyrinthine plot required careful attention to follow all threads.",
            "The cave system was labyrinthine, requiring experienced guides for exploration."
        ]
    },
    'abstraction': {
        'concrete': [
            "She preferred concrete examples rather than abstract theories to explain the concept.",
            "The company wanted concrete results that could be measured and quantified.",
            "The witness provided concrete evidence that contradicted the defendant's story.",
            "They needed concrete plans rather than vague aspirations to move forward.",
            "The experiment provided concrete data supporting the researcher's hypothesis."
        ],
        'tangible': [
            "The project delivered tangible benefits to the community within the first year.",
            "He wanted something tangible to remember his journey, not just digital photos.",
            "The lawyers sought tangible proof before proceeding with the lawsuit.",
            "After years of theoretical work, she wanted to create something tangible.",
            "The museum allowed visitors to have tangible experiences with historical artifacts."
        ],
        'practical': [
            "Her approach was practical, focusing on solutions rather than assigning blame.",
            "The course emphasized practical skills that students could apply immediately.",
            "For practical reasons, they decided to drive rather than fly to the conference.",
            "He offered practical advice based on years of experience in the field.",
            "The practical design prioritized functionality over aesthetic considerations."
        ],
        'balanced': [
            "The news program presented a balanced perspective on the controversial issue.",
            "She took a balanced approach to parenting, providing both structure and freedom.",
            "The ecosystem was perfectly balanced, with predator and prey populations in equilibrium.",
            "The chef created a balanced menu with flavors that complemented each other.",
            "The committee included balanced representation from all stakeholder groups."
        ],
        'theoretical': [
            "The solution was theoretical and had not yet been tested in real-world conditions.",
            "His work remained largely theoretical, based on mathematical models rather than experiments.",
            "The class explored the theoretical foundations of economics before examining case studies.",
            "Her theoretical framework provided new ways to understand social dynamics.",
            "The theoretical physics paper proposed ideas that might not be testable for decades."
        ],
        'abstract': [
            "The painting used abstract shapes and colors to evoke emotional responses.",
            "The philosopher spoke in abstract terms about concepts of justice and freedom.",
            "The music had an abstract quality that defied conventional analysis.",
            "She found it difficult to grasp such abstract concepts without concrete examples.",
            "The sculpture was abstract, representing the idea of tension rather than any physical object."
        ],
        'philosophical': [
            "The novel raised philosophical questions about the nature of consciousness.",
            "They engaged in philosophical debate long into the night, exploring fundamental questions.",
            "His philosophical approach to business prioritized purpose over mere profit.",
            "The documentary examined the philosophical implications of new technologies.",
            "She found philosophical insights in everyday experiences that others overlooked."
        ]
    },
    'composite': {
        'memory': [
            "His memory of childhood summers by the lake remained vivid decades later.",
            "The smell of baking bread triggered a memory of her grandmother's kitchen.",
            "The museum preserved the memory of the disaster for future generations.",
            "Computer memory continues to become both larger and faster with each generation.",
            "Her memory of the conversation differed significantly from his account."
        ],
        'creativity': [
            "The workshop aimed to foster creativity through collaborative exercises.",
            "Her creativity allowed her to see solutions that others missed.",
            "The child's creativity was evident in the imaginative stories she told.",
            "Economic challenges often sparked creativity and innovation in the community.",
            "The artist's creativity seemed unlimited, constantly evolving in new directions."
        ],
        'reasoning': [
            "His reasoning led him to a conclusion that contradicted conventional wisdom.",
            "The detective's reasoning connected seemingly unrelated clues into a coherent theory.",
            "She explained her reasoning step by step so everyone could follow her logic.",
            "The court found flaws in the prosecutor's reasoning and dismissed the case.",
            "Mathematical reasoning requires both precision and intuitive understanding."
        ],
        'intuition': [
            "Her intuition told her something was wrong before any obvious signs appeared.",
            "Experienced traders sometimes relied on intuition as much as market analysis.",
            "The doctor's intuition led him to order tests that revealed the rare condition.",
            "Despite the promising data, his intuition warned him to proceed cautiously.",
            "Artists often describe their creative process as guided by intuition rather than planning."
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
    
    elif pattern == 'concept_leap':
        # Skip a concept in the same space
        if source_idx < len(source_concepts) - 2:
            valid_targets.append((source_space, source_concepts[source_idx + 2]))
        elif source_idx < len(source_concepts) - 1:
            valid_targets.append((source_space, source_concepts[source_idx + 1]))
    
    elif pattern == 'dimensional_shift':
        # Move to a concept in another space
        for other_space in SPACES:
            if other_space != source_space and other_space != 'composite':
                # Try to find similar position in other space
                other_concepts = CONCEPTS[other_space]
                relative_pos = min(source_idx, len(other_concepts) - 1)
                valid_targets.append((other_space, other_concepts[relative_pos]))
    
    elif pattern == 'compositional':
        # Move to a composite concept
        if 'composite' in CONCEPTS:
            for composite_concept in CONCEPTS['composite']:
                valid_targets.append(('composite', composite_concept))
    
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
    source_space = random.choice([s for s in SPACES if s != 'composite'])
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

def main():
    """Generate and save 10,000 samples"""
    samples = []
    
    # Generate samples
    print("Generating 10,000 samples...")
    for i in range(10000):
        sample = generate_sample(i)
        samples.append(sample)
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} samples")
    
    # Save to file
    output_file = "laminet_samples_10k.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_file}")

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
        print(f"  {pattern}: {count} samples ({count/100}%)")
    
    print("\nSource space distribution:")
    for space, count in space_counts["source"].items():
        print(f"  {space}: {count} samples ({count/100}%)")
    
    print("\nTarget space distribution:")
    for space, count in space_counts["target"].items():
        print(f"  {space}: {count} samples ({count/100}%)")

if __name__ == "__main__":
    main() 