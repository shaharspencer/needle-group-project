"""
Which TMDB titles belong to each of our 38 franchises.

Curated by hand rather than resolved by search at runtime. Naive search is not
reliable here: querying "Lord of the Rings" returns a making-of documentary
before the films, "Dune" returns Legally Blonde, and "Fast" returns Fast Getaway.
Franchises with no single collection (the MCU, the DCEU) have to come from a
production company instead. Every id below was checked against the TMDB API
before being written down.

Four ways a franchise can name its titles, and a franchise may use several:
  tv          explicit TV series ids
  collection  TMDB collection ids (film series)
  movie       explicit film ids, for films outside any collection
  company     production company id, for franchises with no collection
"""

FRANCHISE_TITLES: dict[str, dict[str, list[int]]] = {
    # --- TV-only franchises -------------------------------------------------
    "Game of Thrones":        {"tv": [1399]},
    "Boardwalk Empire":       {"tv": [1621]},
    "The Wire":               {"tv": [1438]},
    "The Sopranos":           {"tv": [1398]},
    "Grey's Anatomy":         {"tv": [1416]},
    "Lost":                   {"tv": [4607]},
    "Dexter":                 {"tv": [1405, 259909]},          # + Resurrection
    "Sons of Anarchy":        {"tv": [1409]},
    "Supernatural":           {"tv": [1622]},
    "Spartacus":              {"tv": [46296, 240459]},         # + House of Ashur
    "The 100":                {"tv": [48866]},
    "Vikings":                {"tv": [44217, 116135]},         # + Valhalla
    "Westworld":              {"tv": [63247]},
    "Ozark":                  {"tv": [69740]},
    "Stranger Things":        {"tv": [66732]},
    "The Boys":               {"tv": [76479]},
    "The Walking Dead":       {"tv": [1402, 194583]},          # + Dead City
    "Peaky Blinders":         {"tv": [60574]},
    "Prison Break":           {"tv": [2288]},
    # Breaking Bad (1396) and Better Call Saul (60059) are one franchise in our
    # scrape, plus the El Camino film.
    "Breaking Bad / Better Call Saul": {"tv": [1396, 60059], "movie": [559969]},

    # --- Film franchises ----------------------------------------------------
    "Harry Potter":           {"collection": [1241]},
    "Indiana Jones":          {"collection": [84], "tv": [661]},   # + Young Indy
    "James Bond 007":         {"collection": [645]},
    "Jurassic Park":          {"collection": [328], "tv": [93741, 237512]},
    "Lord of the Rings":      {"collection": [119], "tv": [84773]},  # + Rings of Power
    "Mission: Impossible":    {"collection": [87359]},
    "Pirates of the Caribbean": {"collection": [295]},
    "The Hunger Games":       {"collection": [131635]},
    "The Matrix":             {"collection": [2344]},
    "The Twilight Saga":      {"collection": [33514]},
    "Transformers":           {"collection": [8650]},
    "Fast & Furious":         {"collection": [9485]},
    "Dune":                   {"collection": [726871], "tv": [90228]},  # + Prophecy
    "Avatar":                 {"collection": [87096]},
    # Alien and Predator were scraped as one wiki, so both collections apply.
    "Alien vs Predator":      {"collection": [8091, 399]},
    "Star Wars":              {
        "collection": [10],
        # The main filmed TV series. Star Wars has by far the most non-filmed
        # material, which is exactly why the on-screen filter matters for it.
        "tv": [4194, 60554, 82856, 83867, 105971, 114461, 92830, 203085],
    },

    # --- No collection exists; fall back to the production company ----------
    "Marvel Cinematic Universe": {"company": [420]},    # Marvel Studios
    "DC Extended Universe":      {"company": [128064]}, # DC Films
}


def franchises() -> list[str]:
    return sorted(FRANCHISE_TITLES)
