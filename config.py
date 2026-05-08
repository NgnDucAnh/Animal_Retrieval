import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANIMAL_SOUND_DIR = os.path.join(BASE_DIR, "Animal-Sound")

# AUDIO PROCESSING CONFIG
SR = 22050
DURATION = 5
LOWCUT = 50
HIGHCUT = 10000

# DATABASE CONFIG
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "animal_sounds"
}

# ANIMAL EMOJI MAPPING
ANIMAL_EMOJI = {
    "Aslan": "🦁", "Bear": "🐻", "Cat": "🐱", "Chicken": "🐔",
    "Cow": "🐄", "Dog": "🐶", "Dolphin": "🐬", "Donkey": "🫏",
    "Elephant": "🐘", "Frog": "🐸", "Horse": "🐴", "Monkey": "🐒",
    "Sheep": "🐑", "chirping_birds": "🐦", "crickets": "🦗",
    "crow": "🐦‍⬛", "hen": "🐔", "insects": "🦟", "pig": "🐷",
    "rooster": "🐓",
}
