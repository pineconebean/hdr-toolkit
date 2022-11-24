SCENE_NUMBER = [
    ('Beerfest Lightshow', 122),
    ('Bistro', 293),
    ('Carousel Fireworks', 761),
    ('Car Closeshot', 820),
    ('Car Fullshot', 883),
    ('Cars Longshot', 954),
    ('Fireplace', 1019),
    ('Fishing Closeshot', 1081),
    ('HDR Testimage', 1138),
    ('Poker Fullshot', 1198),
    ('Showgirl 1', 1318),
    ('Showgirl 2', 1363),
    ('Smith Hammering', 1430),
    ('Smith Welding', 1494)
]


def get_scene(sample_number):
    for scene, number in SCENE_NUMBER:
        if sample_number < number:
            return scene
    raise ValueError('invalid sample number (not in range [0, 1494))')


def num_scene_samples(percent=False):
    result = [SCENE_NUMBER[0]]
    for i in range(1, len(SCENE_NUMBER)):
        result.append((SCENE_NUMBER[i][0], SCENE_NUMBER[i][1] - SCENE_NUMBER[i - 1][1]))
    if percent:
        result = list(map(lambda x: (x[0], x[1] / SCENE_NUMBER[-1][1]), result))
    return result
