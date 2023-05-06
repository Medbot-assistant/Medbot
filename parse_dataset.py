import os
from lxml import etree

# Function to parse an XML file and return the etree object
def parse_xml_file(file_path):
    return etree.parse(file_path)

# Change the directory to Dataset
rootdir = './MedQuAD'

# Initialize an empty list to store the XML trees
xml_trees = []

# Iterate through the files in the Dataset directory
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # Check if the file is an XML file
        if file.endswith('.xml'):
            # Parse the XML file and add the etree object to the list
            filepath = os.path.join(subdir, file)
            xml_trees.append(parse_xml_file(filepath))

# Print the number of XML files read
print(f"Read {len(xml_trees)} XML files.")

class Document:
    def __init__(self, id, source, url, focus, umls, qa_pairs):
        self.id = id
        self.source = source
        self.url = url
        self.focus = focus
        self.umls = umls
        self.qa_pairs = qa_pairs

class UMLS:
    def __init__(self, cuis, semantic_types, semantic_group):
        self.cuis = cuis
        self.semantic_types = semantic_types
        self.semantic_group = semantic_group

class QAPair:
    def __init__(self, pid, question, answer):
        self.pid = pid
        self.question = question
        self.answer = answer

class Question:
    def __init__(self, qid, qtype, text):
        self.qid = qid
        self.qtype = qtype
        self.text = text

class Answer:
    def __init__(self, text):
        self.text = text

def parse_document(xml_tree):
    root = xml_tree.getroot()

    id = root.get("id")
    source = root.get("source")
    url = root.get("url")
    try:
        focus = root.find("Focus").text
    except AttributeError:
        focus = None

    #umls_elem = root.find("FocusAnnotations/UMLS")
    #cuis = [cui.text for cui in umls_elem.findall("CUIs/CUI")]
    #semantic_types = [st.text for st in umls_elem.findall("SemanticTypes/SemanticType")]
    #semantic_group = umls_elem.find("SemanticGroup").text
    #umls = UMLS(cuis, semantic_types, semantic_group)

    qa_pairs = []
    for qa_pair_elem in root.findall("QAPairs/QAPair"):
        pid = qa_pair_elem.get("pid")

        question_elem = qa_pair_elem.find("Question")
        qid = question_elem.get("qid")
        qtype = question_elem.get("qtype")
        question_text = question_elem.text
        question = Question(qid, qtype, question_text)

        answer_text = qa_pair_elem.find("Answer").text
        answer = Answer(answer_text)

        qa_pairs.append(QAPair(pid, question, answer))

    return Document(id, source, url, focus, None, qa_pairs)

import pandas as pd

# Function to convert a Document object to a dictionary
def document_to_dict(document):
    return {
        "id": document.id,
        "source": document.source,
        "url": document.url,
        "focus": document.focus,
  #      "umls_cuis": ','.join(document.umls.cuis),
        #"umls_semantic_types": ','.join(document.umls.semantic_types),
        #"umls_semantic_group": document.umls.semantic_group,
        "qa_pairs": document.qa_pairs,
    }

# Parse all XML trees and convert Document objects to dictionaries
document_dicts = [document_to_dict(parse_document(xml_tree)) for xml_tree in xml_trees]

# Create a pandas DataFrame from the list of dictionaries
df = pd.DataFrame(document_dicts)

df.to_csv('./dataset.tsv', sep="\t")