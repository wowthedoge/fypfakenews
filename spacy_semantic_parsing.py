import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Your sentence
sentence = "I do my work in the library today."
doc = nlp(sentence)

# Initialize lists for subjects, verbs, objects, adverbials, and prepositional phrases
subjects, verbs, direct_objects, adverbials, prepositional_phrases = [], [], [], [], []

# Initialize variables for time and location
time, location = None, None

# Process the sentence for dependency parsing
for token in doc:
    # Identifying subjects
    if "subj" in token.dep_:
        subjects.append(token.text)
    # Identifying verbs
    elif token.pos_ == "VERB":
        verbs.append(token.lemma_)
    # Identifying direct objects
    elif "obj" in token.dep_:
        direct_objects.append(token.text)
    # Identifying adverbial modifiers of the verb
    elif token.dep_ == "advmod":
        adverbials.append(token.text)
    # Identifying prepositional phrases
    elif token.dep_ in ["prep", "pobj"]:
        prepositional_phrases.append(token.text)

# Process the sentence for named entity recognition
for ent in doc.ents:
    if ent.label_ == "TIME":
        time = ent.text
    elif ent.label_ == "GPE" or ent.label_ == "LOC":
        location = ent.text

# Combine lists for output
subjects_text = ', '.join(subjects)
verbs_text = ', '.join(verbs)
direct_objects_text = ', '.join(direct_objects)
adverbial_text = ', '.join(adverbials)
prepositional_phrase_text = ', '.join(prepositional_phrases)

# Print results
print(f"Subjects: {subjects_text}")
print(f"Verbs: {verbs_text}")
print(f"Direct Objects: {direct_objects_text}")
print(f"Adverbials: {adverbial_text}")
print(f"Prepositional Phrases: {prepositional_phrase_text}")
print(f"Time: {time}")
print(f"Location: {location}")
