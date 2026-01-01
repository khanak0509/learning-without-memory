import re 


def extract_target_words(user_input):
    match = re.search(r'(\d+)\s*words?', user_input.lower())
    if match:
        return int(match.group(1))
    return None

# print(extract_target_words("sumarise in 100 word"))
