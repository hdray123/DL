action=['up','left','down','right','resstart','exit']
letter_codes=[ord(ch) for ch in "WASDRQwasdrq"]
action_dict=dict(zip(letter_codes,action*2))
print(action_dict)