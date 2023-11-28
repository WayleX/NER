import spacy
def find_mountain_name(sentence:str):
    nlp_ner = spacy.load("./model-best")

    doc = nlp_ner(sentence)

    colors = {"MOUNTAIN":"#FFFFFF"}
    options = {"colors": colors} 

    spacy.displacy.render(doc, style="ent", options= options, jupyter=True)