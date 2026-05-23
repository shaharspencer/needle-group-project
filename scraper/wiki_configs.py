"""
wiki_configs.py
---------------
Per-wiki configuration for the Fandom character death scraper.

Only PG-13+ franchises are included to ensure deaths are literal,
plot-relevant fatalities rather than metaphorical or off-screen events.
"""

WIKI_CONFIGS = {
    "mcu": {
        "name": "MCU",
        "franchise": "Marvel Cinematic Universe",
        "base_url": "https://marvelcinematicuniverse.fandom.com",
        "api_url": "https://marvelcinematicuniverse.fandom.com/api.php",
        "content_rating": "PG-13",
        "character_categories": ["Marvel Cinematic Universe Characters", "Characters"],
        "infobox_templates": ["Character"],
        "field_map": {
            "real name": "real_name", "status": "status", "species": "species", 
            "gender": "gender", "dob": "dob", "dod": "dod", "actor": "actor", 
            "affiliation": "affiliation", "movie": "appears_in_movies", 
            "tv series": "appears_in_tv", "citizenship": "citizenship"
        },
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "disintegrated", "deleted"],
        "alive_statuses": ["alive", "living", "active", "alive (resurrected)", "unknown"],
    },

    "got": {
        "name": "GoT",
        "franchise": "Game of Thrones",
        "base_url": "https://gameofthrones.fandom.com",
        "api_url": "https://gameofthrones.fandom.com/api.php",
        "content_rating": "TV-MA",
        "character_categories": ["Individuals", "Characters"],
        "infobox_templates": ["Character", "Infobox Character"],
        "dod_only": True,
        "field_map": {
            "name": "real_name", "status": "status", "species": "species",
            "gender": "gender", "birth": "dob", "death": "dod", "actor": "actor",
            "allegiance": "affiliation", "culture": "citizenship", "seasons": "appears_in_movies"
        },
        "dead_statuses": ["deceased", "dead", "killed", "executed", "burned", "beheaded", "poisoned", "stabbed", "destroyed"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "walking_dead": {
        "name": "TWD",
        "franchise": "The Walking Dead",
        "base_url": "https://walkingdead.fandom.com",
        "api_url": "https://walkingdead.fandom.com/api.php",
        "content_rating": "TV-MA",
        "character_categories": ["Characters", "TV Series Characters"],
        "infobox_templates": [
            "Character", "Infobox", "TV Character", "TV Character Infobox",
            "Comic Character", "Video Game Character", "Fear Character",
        ],
        "field_map": {
            "name": "real_name", "status": "status", "species": "species",
            "gender": "gender", "born": "dob", "death": "dod",
            "portrayed": "actor", "portrayed by": "actor",
            "affiliation": "affiliation", "nationality": "citizenship",
            "seasons": "appears_in_movies", "occupation": "title",
            "ethnicity": "species",
        },
        "dead_statuses": ["deceased", "dead", "killed", "devoured", "undead", "destroyed", "reanimated", "put down"],
        "alive_statuses": ["alive", "alive (off-screen)", "unknown"],
    },

    "star_wars": {
        "name": "StarWars",
        "franchise": "Star Wars",
        "base_url": "https://starwars.fandom.com",
        "api_url": "https://starwars.fandom.com/api.php",
        "content_rating": "PG-13",
        "character_categories": ["Individuals", "Characters"],
        "infobox_templates": ["Character", "Individual"],
        "dod_only": True,
        "field_map": {
            "name": "real_name", "status": "status", "species": "species",
            "gender": "gender", "birth": "dob", "death": "dod", "actor": "actor",
            "affiliation": "affiliation", "homeworld": "citizenship", "films": "appears_in_movies"
        },
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "became one with the force", "disintegrated"],
        "alive_statuses": ["alive", "active", "unknown"],
    },

    "harry_potter": {
        "name": "HP",
        "franchise": "Harry Potter",
        "base_url": "https://harrypotter.fandom.com",
        "api_url": "https://harrypotter.fandom.com/api.php",
        "content_rating": "PG-13",
        "character_categories": ["Individuals", "Characters"],
        "infobox_templates": ["Individual infobox", "Character"],
        "dod_only": True,
        "field_map": {
            "name": "real_name", "status": "status", "species": "species",
            "gender": "gender", "born": "dob", "died": "dod", "actor": "actor",
            "affiliation": "affiliation", "nationality": "citizenship", "films": "appears_in_movies"
        },
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "destroyed", "killed by voldemort"],
        "alive_statuses": ["alive", "living", "unknown"],
    },

    "hunger_games": {
        "name": "HungerGames",
        "franchise": "The Hunger Games",
        "base_url": "https://thehungergames.fandom.com",
        "api_url": "https://thehungergames.fandom.com/api.php",
        "content_rating": "PG-13",
        "character_categories": ["Characters"],
        "infobox_templates": ["Character"],
        "field_map": {
            "name": "real_name", "fate": "status", "status": "status", "species": "species", 
            "gender": "gender", "age": "dob", "actor": "actor", 
            "home": "citizenship", "movieappears": "appears_in_movies"
        },
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "executed", "died"],
        "alive_statuses": ["alive", "living", "unknown"],
    },

    "lotr": {
        "name": "LOTR",
        "franchise": "Lord of the Rings",
        "base_url": "https://lotr.fandom.com",
        "api_url": "https://lotr.fandom.com/api.php",
        "content_rating": "PG-13",
        "character_categories": ["Characters", "Characters by race"],
        "infobox_templates": ["Infobox Person", "Infobox Person Hobbits", "Infobox Person Men", "Infobox Person Elves", "Infobox Person Dwarves", "Infobox Person Maiar", "Character"],
        "dod_only": True,
        "field_map": {
            "name": "real_name", "status": "status", "race": "species",
            "gender": "gender", "birth": "dob", "death": "dod", "actor": "actor",
            "culture": "citizenship"
        },
        "dead_statuses": ["deceased", "dead", "killed", "slain", "destroyed", "died"],
        "alive_statuses": ["alive", "living", "departed", "unknown"],
    },

    "avatar": {
        "name": "Avatar",
        "franchise": "Avatar",
        "base_url": "https://james-camerons-avatar.fandom.com",
        "api_url": "https://james-camerons-avatar.fandom.com/api.php",
        "content_rating": "PG-13",
        "character_categories": ["Characters", "Film characters"],
        "infobox_templates": ["Infobox character", "Character"],
        "field_map": {
            "name": "real_name", "status": "status", "species": "species", 
            "gender": "gender", "born": "dob", "died": "dod", "actor": "actor", 
            "clan": "affiliation"
        },
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "died"],
        "alive_statuses": ["alive", "living", "unknown"],
    },
    "twilight": {
        "name": "Twilight",
        "franchise": "The Twilight Saga",
        "base_url": "https://twilightsaga.fandom.com",
        "api_url": "https://twilightsaga.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Vampire infobox", "Human infobox", "Werewolf infobox", "Shape-shifter infobox"],
        "dod_only": True,
        "field_map": {"status": "status", "born": "dob", "died": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed"],
        "alive_statuses": ["alive", "living", "undead", "vampire", "unknown"],
    },

    "matrix": {
        "name": "Matrix",
        "franchise": "The Matrix",
        "base_url": "https://matrix.fandom.com",
        "api_url": "https://matrix.fandom.com/api.php",
        "content_rating": "R",
        "infobox_templates": [
            "Infobox character", "Character", "Infobox Character",
            "CharacterInfobox", "Character infobox", "Infobox/Character",
            "Infobox person", "Person", "Human", "Agent", "Program",
        ],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod",
            "gender": "gender", "portrayed by": "actor", "portrayed": "actor",
            "affiliation": "affiliation", "occupation": "title",
        },
        "dead_statuses": ["deceased", "dead", "killed", "deleted", "destroyed"],
        "alive_statuses": ["alive", "living", "active", "activated", "unknown"],
    },

    "indiana_jones": {
        "name": "IndianaJones",
        "franchise": "Indiana Jones",
        "base_url": "https://indianajones.fandom.com",
        "api_url": "https://indianajones.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Infobox Character", "Character"],
        "dod_only": True,
        "field_map": {"status": "status", "birth": "dob", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed"],
        "alive_statuses": ["alive", "living", "unknown"],
    },

    "jurassic_park": {
        "name": "JurassicPark",
        "franchise": "Jurassic Park",
        "base_url": "https://jurassicpark.fandom.com",
        "api_url": "https://jurassicpark.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Character", "Dinosaur"],
        "field_map": {"status": "status", "born": "dob", "died": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "eaten"],
        "alive_statuses": ["alive", "living", "unknown"],
    },

    "avp": {
        "name": "AVP",
        "franchise": "Alien vs Predator",
        "base_url": "https://avp.fandom.com",
        "api_url": "https://avp.fandom.com/api.php",
        "content_rating": "R",
        "infobox_templates": ["Infobox Character", "Xenomorph", "Yautja"],
        "field_map": {"status": "status", "birth": "dob", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "impregnated", "chestbursted"],
        "alive_statuses": ["alive", "living", "unknown"],
    },

    "fast_furious": {
        "name": "FastFurious",
        "franchise": "Fast & Furious",
        "base_url": "https://fastandfurious.fandom.com",
        "api_url": "https://fastandfurious.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Character infobox", "Character"],
        "field_map": {"status": "status", "birth": "dob", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed"],
        "alive_statuses": ["alive", "living", "unknown"],
    },

    "pirates": {
        "name": "Pirates",
        "franchise": "Pirates of the Caribbean",
        "base_url": "https://pirates.fandom.com",
        "api_url": "https://pirates.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Character", "Infobox character"],
        "dod_only": True,
        "field_map": {"status": "status", "birth": "dob", "death": "dod", "born": "dob", "died": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "executed"],
        "alive_statuses": ["alive", "living", "undead", "unknown"],
    },

    "transformers": {
        "name": "Transformers",
        "franchise": "Transformers",
        "base_url": "https://transformers.fandom.com",
        "api_url": "https://transformers.fandom.com/api.php",
        "content_rating": "PG-13",
        # Cast a wide net for the many template variants used on this wiki
        "infobox_templates": ["Character", "Infobox character", "Infobox Character",
                              "CharBox", "CharInfobox", "Cybertronian", "TransformerInfobox"],
        "field_map": {"status": "status", "birth": "dob", "death": "dod", "born": "dob", "died": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "offline", "deactivated"],
        "alive_statuses": ["alive", "living", "online", "active", "unknown"],
    },

    "dune": {
        "name": "Dune",
        "franchise": "Dune",
        "base_url": "https://dune.fandom.com",
        "api_url": "https://dune.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Character", "Infobox character"],
        "dod_only": True,
        "field_map": {"status": "status", "birth": "dob", "death": "dod", "born": "dob", "died": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed"],
        "alive_statuses": ["alive", "living", "unknown"],
    },
    # NOTE: The first walking_dead entry (lines 46-61) has the full field_map;
    # this duplicate was retained for template name coverage but is now removed.

    "dceu": {
        "name": "DCEU",
        "franchise": "DC Extended Universe",
        "base_url": "https://dcextendeduniverse.fandom.com",
        "api_url": "https://dcextendeduniverse.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Character", "Infobox character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "destroyed"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },
    "breaking_bad": {
        "name": "BreakingBad",
        "franchise": "Breaking Bad / Better Call Saul",
        "base_url": "https://breakingbad.fandom.com",
        "api_url": "https://breakingbad.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Infobox Character2", "Character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "executed", "dissolved"],
        "alive_statuses": ["alive", "living", "unknown"],
    },

    "the_boys": {
        "name": "TheBoys",
        "franchise": "The Boys",
        "base_url": "https://the-boys.fandom.com",
        "api_url": "https://the-boys.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Character", "Infobox character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "exploded", "lasered"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "stranger_things": {
        "name": "StrangerThings",
        "franchise": "Stranger Things",
        "base_url": "https://strangerthings.fandom.com",
        "api_url": "https://strangerthings.fandom.com/api.php",
        "content_rating": "TV-14",
        "infobox_templates": ["Character", "Infobox character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "disintegrated"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "supernatural": {
        "name": "Supernatural",
        "franchise": "Supernatural",
        "base_url": "https://supernatural.fandom.com",
        "api_url": "https://supernatural.fandom.com/api.php",
        "content_rating": "TV-14",
        "infobox_templates": ["RecurringCharacters"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "alias": "real_name", "category": "species", "species": "species"
        },
        "dead_statuses": ["deceased", "dead", "killed", "slain", "destroyed"],
        "alive_statuses": ["alive", "living", "active", "resurrected", "unknown"],
    },
    "james_bond": {
        "name": "JamesBond",
        "franchise": "James Bond 007",
        "base_url": "https://jamesbond.fandom.com",
        "api_url": "https://jamesbond.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Character", "Infobox character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "shot", "exploded", "eaten"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "mission_impossible": {
        "name": "MissionImpossible",
        "franchise": "Mission: Impossible",
        "base_url": "https://missionimpossible.fandom.com",
        "api_url": "https://missionimpossible.fandom.com/api.php",
        "content_rating": "PG-13",
        "infobox_templates": ["Infobox character", "Character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "vikings": {
        "name": "Vikings",
        "franchise": "Vikings",
        "base_url": "https://vikings.fandom.com",
        "api_url": "https://vikings.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Infobox character", "Character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "slain", "beheaded", "burned", "sacrificed", "blood eagled"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "sons_of_anarchy": {
        "name": "SonsOfAnarchy",
        "franchise": "Sons of Anarchy",
        "base_url": "https://sonsofanarchy.fandom.com",
        "api_url": "https://sonsofanarchy.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Infobox character", "Character"],
        "field_map": {"status": "status", "born": "dob", "died": "dod", "death": "dod"},
        "dead_statuses": ["deceased", "dead", "killed", "executed", "murdered"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "spartacus": {
        "name": "Spartacus",
        "franchise": "Spartacus",
        "base_url": "https://spartacus.fandom.com",
        "api_url": "https://spartacus.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Infobox character", "Character Template"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality", "allegiance": "affiliation",
        },
        "dead_statuses": ["deceased", "dead", "killed", "slain", "executed", "crucified",
                          "murdered", "stabbed", "burned", "beheaded"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "westworld": {
        "name": "Westworld",
        "franchise": "Westworld",
        "base_url": "https://westworld.fandom.com",
        "api_url": "https://westworld.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Infobox/Character", "Infobox/Host Character - old"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "species": "species", "affiliation": "affiliation",
            "actor": "actor",
        },
        "dead_statuses": ["deceased", "dead", "killed", "destroyed", "decommissioned",
                          "wiped", "deleted", "retired"],
        "alive_statuses": ["alive", "active", "operational", "online", "unknown"],
    },

    "sopranos": {
        "name": "Sopranos",
        "franchise": "The Sopranos",
        "base_url": "https://thesopranos.fandom.com",
        "api_url": "https://thesopranos.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Character infobox", "Character infobox v2", "InfoboxCharacter"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality", "affiliation": "affiliation",
            "actor": "actor",
        },
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "whacked",
                          "executed", "shot", "strangled"],
        "alive_statuses": ["alive", "living", "active", "imprisoned", "unknown"],
    },

    "boardwalk": {
        "name": "BoardwalkEmpire",
        "franchise": "Boardwalk Empire",
        "base_url": "https://boardwalkempire.fandom.com",
        "api_url": "https://boardwalkempire.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Character infobox", "Infobox character"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality", "affiliation": "affiliation",
            "actor": "actor",
        },
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "executed", "shot"],
        "alive_statuses": ["alive", "living", "active", "imprisoned", "unknown"],
    },

    "lost": {
        "name": "Lost",
        "franchise": "Lost",
        "base_url": "https://lostpedia.fandom.com",
        "api_url": "https://lostpedia.fandom.com/api.php",
        "content_rating": "TV-14",
        "infobox_templates": ["Infobox Character"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality", "affiliation": "affiliation",
            "actor": "actor",
        },
        "dead_statuses": ["deceased", "dead", "killed", "sacrificed", "executed",
                          "murdered", "drowned", "shot", "blown up"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "the_100": {
        "name": "The100",
        "franchise": "The 100",
        "base_url": "https://the100.fandom.com",
        "api_url": "https://the100.fandom.com/api.php",
        "content_rating": "TV-14",
        "infobox_templates": ["Character"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality", "affiliation": "affiliation",
            "actor": "actor", "species": "species",
        },
        "dead_statuses": ["deceased", "dead", "killed", "executed", "transcended",
                          "murdered", "shot", "blown up"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "the_wire": {
        "name": "TheWire",
        "franchise": "The Wire",
        "base_url": "https://thewire.fandom.com",
        "api_url": "https://thewire.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Infobox Character"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality", "affiliation": "affiliation",
            "actor": "actor",
        },
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "shot", "executed"],
        "alive_statuses": ["alive", "living", "active", "imprisoned", "unknown"],
    },

    # -------------------------------------------------------------------------
    # New franchises added to increase data diversity
    # -------------------------------------------------------------------------

    "greys_anatomy": {
        "name": "GreysAnatomy",
        "franchise": "Grey's Anatomy",
        "base_url": "https://greysanatomy.fandom.com",
        "api_url": "https://greysanatomy.fandom.com/api.php",
        "content_rating": "TV-14",
        "infobox_templates": [
            "Attending Infobox", "Intern and Resident Infobox",
            "Firefighter Infobox", "Character Family Infobox",
        ],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality",
            "portrayed by": "actor", "actor": "actor", "portrayed": "actor",
            "occupation": "title", "specialty": "title",
            "affiliation": "affiliation", "name": "real_name",
        },
        "dead_statuses": ["deceased", "dead", "killed", "died", "passed away"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },

    "peaky_blinders": {
        "name": "PeakyBlinders",
        "franchise": "Peaky Blinders",
        "base_url": "https://peakyblinders.fandom.com",
        "api_url": "https://peakyblinders.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Character", "Infobox character", "Infobox Character"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality",
            "portrayed by": "actor", "actor": "actor", "portrayed": "actor",
            "occupation": "title", "affiliation": "affiliation",
            "allegiance": "affiliation",
        },
        "dead_statuses": ["deceased", "dead", "killed", "executed", "murdered", "shot"],
        "alive_statuses": ["alive", "living", "active", "imprisoned", "unknown"],
    },

    "dexter": {
        "name": "Dexter",
        "franchise": "Dexter",
        "base_url": "https://dexter.fandom.com",
        "api_url": "https://dexter.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["DualProfile"],
        "field_map": {
            "status": "status", "birth_date": "dob", "born": "dob",
            "gender": "gender", "actor": "actor",
            "full name": "real_name", "name": "real_name",
            "profession": "title", "affiliation": "affiliation",
            "ethnicity": "species",
        },
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "executed"],
        "alive_statuses": ["alive", "living", "active", "imprisoned", "unknown"],
    },

    "prison_break": {
        "name": "PrisonBreak",
        "franchise": "Prison Break",
        "base_url": "https://prisonbreak.fandom.com",
        "api_url": "https://prisonbreak.fandom.com/api.php",
        "content_rating": "TV-14",
        "infobox_templates": ["Character", "Infobox character", "Infobox Character", "Char"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality",
            "portrayed by": "actor", "actor": "actor", "portrayed": "actor",
            "occupation": "title", "affiliation": "affiliation",
        },
        "dead_statuses": ["deceased", "dead", "killed", "executed", "murdered"],
        "alive_statuses": ["alive", "living", "active", "imprisoned", "unknown"],
    },

    "ozark": {
        "name": "Ozark",
        "franchise": "Ozark",
        "base_url": "https://ozark.fandom.com",
        "api_url": "https://ozark.fandom.com/api.php",
        "content_rating": "TV-MA",
        "infobox_templates": ["Character", "Infobox character", "Infobox Character"],
        "field_map": {
            "status": "status", "born": "dob", "died": "dod", "death": "dod",
            "gender": "gender", "nationality": "nationality",
            "portrayed by": "actor", "actor": "actor", "portrayed": "actor",
            "occupation": "title", "affiliation": "affiliation",
        },
        "dead_statuses": ["deceased", "dead", "killed", "murdered", "executed"],
        "alive_statuses": ["alive", "living", "active", "unknown"],
    },
}
