import random

def answer_to_life():
    return random.choice(life_answers)

def answer_prompt(question_prompt: str) -> str:
    return random.choice(answer_prompts[str(question_prompt)])

# language model prompts/seeds
answer_prompts = {
    'learning': [
        'The best way to learn anything is ',
        'If you want to learn anything new, I suggest you ',
        'The most effective learning approach is to ',
        'If you want to learn fast, I suggest ',
    ],
    'habits': [
        'Some people may find this habit odd, but ',
        'One unusual habit I have is ',
        'My habits may seem odd, but they work for me. For example, my habit of ',
        'Habits are important for success. My favorite habit is '
    ],
    'beliefs': [
        'Nearly nobody agrees with me on this, but I believe ',
        'I think most people would disagree with me on this, but I believe ',
        'People rarely agree with me on this, but I firmly believe ',
        'Some may disagree with this, but I strongly believe '
    ],
    'stress': [
        'When I feel overwhelmed, what helps me is ',
        "If I'm feeling anxious, I take a deep breath and tell myself ",
        'If I feel overwhelmed and want to calm myself down, I like to  ',
        'If I feel un-focused or anxious, I calm myself down by '
    ],
    'purpose': [
        'The purpose of life is ',
        'I think the purpose of life is ',
        'For me, the purpose of life is ',
        "The purpose of one's life should be "
    ]    
}

# format linked text
hyperlink_format = '<a href="{website}" style="color:blue; border-bottom: 1px solid" target="_blank">{text}</a>'
google_forty_two = hyperlink_format.format(website='https://www.google.com/search?q=the+answer+to+life+the+universe+and+everything', text='42')

# "42" anwers
life_answers = [
    "Hmm, have to think about that. Return to this place in exactly 7 and a half million years.<br><br>#DeepThought 🤔🤖",
    "You're not going to like it...the answer to the ultimate question of Life, the Universe, and Everything is...</br></br>"+google_forty_two+" 🤔🤖",
    google_forty_two,
]

