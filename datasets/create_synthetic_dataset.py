import random
import json

themes = ["Quest", "Science", "Love Story", "Mystery", "Revenge", "Discovery"]

def make_story(theme):
    templates = {
        "Quest": "The {hero} ventured into the {setting} seeking the {goal}.",
        "Science": "The {scientist} studied the {phenomenon} in the {lab}.",
        "Love Story": "The {person1} met the {person2} during the {event}.",
        "Mystery": "The {detective} uncovered clues at the {location}.",
        "Revenge": "The {avenger} sought justice against the {enemy}.",
        "Discovery": "The {explorer} uncovered the {artifact} in the {setting}.",
    }
    fill = {
        "hero": random.choice(["knight", "explorer", "pilgrim"]),
        "setting": random.choice(["ancient forest", "ruins", "desert"]),
        "goal": random.choice(["lost crown", "hidden treasure", "secret scroll"]),
        "scientist": random.choice(["physicist", "biologist", "astronomer"]),
        "phenomenon": random.choice(["black hole", "genetic anomaly", "time ripple"]),
        "lab": random.choice(["deep lab", "hidden base", "floating observatory"]),
        "person1": random.choice(["artist", "traveler", "scholar"]),
        "person2": random.choice(["dancer", "wanderer", "lost soul"]),
        "event": random.choice(["storm", "festival", "eclipse"]),
        "detective": random.choice(["investigator", "detective", "inspector"]),
        "location": random.choice(["mansion", "museum", "sewer"]),
        "avenger": random.choice(["warrior", "vigilante", "hero"]),
        "enemy": random.choice(["tyrant", "betrayer", "rival"]),
        "explorer": random.choice(["archaeologist", "traveler", "scientist"]),
        "artifact": random.choice(["ancient map", "cursed stone", "golden idol"]),
    }
    story = templates[theme].format(**fill)
    return {"text": story, "label": theme}

data = [make_story(random.choice(themes)) for _ in range(10000)]

with open("laminet_synthetic_dataset.json", "w") as f:
    json.dump(data, f)
