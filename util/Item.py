class Item:
    question = ""
    subject = ""
    relation = ""
    obj = ""
    questionVector = None
    candidatePool = []#candidatePool key: subject id value: list of relation id

    def __init__(self, question, sub, relation, obj,candidatePool):
        self.question = question
        self.subject = sub
        self.relation = relation
        self.obj = obj
        self.candidatePool = candidatePool
