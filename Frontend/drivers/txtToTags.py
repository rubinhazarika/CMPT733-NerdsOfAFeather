import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def txtToTags(caption):
    # Tokenize the caption
    tokens = word_tokenize(caption)

    # Tag each token with its part of speech
    tagged_tokens = pos_tag(tokens)

    # Extract specific tokens based on their part of speech
    nouns = [token[0] for token in tagged_tokens if token[1].startswith("N")]
    verbs = [token[0] for token in tagged_tokens if token[1].startswith("V")]
    adjectives = [token[0] for token in tagged_tokens if token[1].startswith("J")]

    # Combine extracted tokens into meaningful phrases
    noun_phrases = []
    current_phrase = []
    for token in tagged_tokens:
        if token[1].startswith("N"):
            current_phrase.append(token[0])
        elif current_phrase:
            # Combine consecutive nouns into noun phrases
            noun_phrases.append(" ".join(current_phrase))
            current_phrase = []
    if current_phrase:
        noun_phrases.append(" ".join(current_phrase))

    verb_phrases = verbs

    adjective_phrases = []
    current_phrase = []
    for token in tagged_tokens:
        if token[1].startswith("J"):
            current_phrase.append(token[0])
        elif current_phrase:
            # Combine consecutive adjectives into adjective phrases
            adjective_phrases.append(" ".join(current_phrase))
            current_phrase = []
    if current_phrase:
        adjective_phrases.append(" ".join(current_phrase))

   
    extracted_tokens =  verb_phrases + adjective_phrases + noun_phrases 

    return extracted_tokens

