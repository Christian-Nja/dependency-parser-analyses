import pyconll as pcl
import io
import matplotlib.pyplot as plt
import argparse
import sys
from conll18_ud_eval import *
from collections import defaultdict
from copy import copy
import unittest
from collections import OrderedDict


FUNCTIONAL_DEPRELS = {
    "aux", "cop", "mark", "det", "clf", "case", "cc", "nsubj", "obj", "iobj", 
    "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
    "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
    "parataxis", "orphan", "goeswith", "reparandum", "root", "dep", "punct"
}

def _decode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, str) else text.decode("utf-8")

def _encode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, unicode) else text.encode("utf-8")
    
def load_conllu(file):
    # Internal representation classes
    class UDRepresentation:
        def __init__(self):
            # Characters of all the tokens in the whole file.
            # Whitespace between tokens is not included.
            self.characters = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.tokens = []
            # List of UDWord instances.
            self.words = []
            # List of UDSpan instances with start&end indices into `characters`.
            self.sentences = []
    class UDSpan:				# i tokens
        def __init__(self, start, end):
            self.start = start
            # Note that self.end marks the first position **after the end** of span,
            # so we can use characters[start:end] or range(start, end).
            self.end = end
    class UDWord:		# c'è ereditarietà. Lo span di word è un oggetto. se faccio .start 
        def __init__(self, span, columns, is_multiword):
            # Span of this word (or MWT, see below) within ud_representation.characters.
            self.span = span
            # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
            self.columns = columns
            # is_multiword==True means that this word is part of a multi-word token.
            # In that case, self.span marks the span of the whole multi-word token.
            self.is_multiword = is_multiword
            # Reference to the UDWord instance representing the HEAD (or None if root).
            self.parent = None
            # List of references to UDWord instances representing functional-deprel children.
            self.functional_children = []
            # Only consider universal FEATS.
            self.columns[FEATS] = "|".join(sorted(feat for feat in columns[FEATS].split("|")
                                                  if feat.split("=", 1)[0] in UNIVERSAL_FEATURES))
            # Let's ignore language-specific deprel subtypes.
            self.columns[DEPREL] = columns[DEPREL].split(":")[0]
            # Precompute which deprels are CONTENT_DEPRELS and which FUNCTIONAL_DEPRELS
            self.is_content_deprel = self.columns[DEPREL] in CONTENT_DEPRELS
            self.is_functional_deprel = self.columns[DEPREL] in FUNCTIONAL_DEPRELS
            # Length of sentence which the word belongs to
            self.sentence_length = None
            # UID unique identifier
            self.uid = None
            # non_projective_grade of a dependency arc from a word w to a word u is the number of words occuring between w and u
            # not directly discending from w and modifying a word not occurring between w and u.
            self.non_projective_grade = 0    
				
    ud = UDRepresentation()

    # Load the CoNLL-U file
    index, uid, sentence_start = 0, 0, None
    while True:
        line = file.readline()
        if not line:
            break
        line = _decode(line.rstrip("\r\n"))

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:
            # Add parent and children UDWord links and check there are no cycles
            def process_word(word):
                if word.parent == "remapping":
                    raise UDError("There is a cycle in a sentence")
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head < 0 or head > len(ud.words) - sentence_start:
                        raise UDError("HEAD '{}' points outside of the sentence".format(_encode(word.columns[HEAD])))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent

            for word in ud.words[sentence_start:]:
                # add to any UDWord the length of sentence
                if word.sentence_length is None: word.sentence_length = len(ud.words[sentence_start:])
                else: raise UDError("Word '{}' has already a sentence length assigned".format(_encode(word.columns[FORM])))
                # another check to sentence length
                if len(ud.words[sentence_start:]) != int(ud.words[-1].columns[ID]): raise UDError("The sentence length is different from sentence's last word ID") 
                # add uid (unique identifier), uid is added after UDWord creation to avoid multi-word bugs
                word.uid = uid
                uid += 1
                process_word(word)
            # func_children cannot be assigned within process_word
            # because it is called recursively and may result in adding one child twice.
            for word in ud.words[sentence_start:]:
                if word.parent and word.is_functional_deprel:
                    word.parent.functional_children.append(word)

                # we add here non projective degree 		                
                if word.parent:
                    i = word.uid # i word wi index
                    j = word.parent.uid # j word wj index	   
                    if i < j:
                        # iterate through span of a word and it's dependency                      	
                    	   for w in ud.words[i+1:j]:
                    	       if w.parent:
                    	           z = int(w.parent.uid) # z word wz in the span index
                    	           if (z < i) or (z > j): word.non_projective_grade += 1	 
                    else:
                    		for w in ud.words[j+1:i]:
                    		    if w.parent:
                    		        z = int(w.parent.uid)
                    		        if (z > i) or (z < j): word.non_projective_grade += 1
                    		                               		             	   
            # Check there is a single root node
            if len([word for word in ud.words[sentence_start:] if word.parent is None]) != 1:
                raise UDError("There are multiple roots in a sentence")

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line.split("\t")
        if len(columns) != 10:
            raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(_encode(line)))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM, so gold.characters == system.characters
        # even if one of them tokenizes the space. Use any Unicode character
        # with category Zs.
        columns[FORM] = "".join(filter(lambda c: unicodedata.category(c) != "Zs", columns[FORM]))
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            try:
                start, end = map(int, columns[ID].split("-"))
            except:
                raise UDError("Cannot parse multi-word token ID '{}'".format(_encode(columns[ID])))

            for _ in range(start, end + 1):
                word_line = _decode(file.readline().rstrip("\r\n"))
                word_columns = word_line.split("\t")
                if len(word_columns) != 10:
                    raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(_encode(word_line)))
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID '{}'".format(_encode(columns[ID])))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected '{}'".format(
                    _encode(columns[ID]), _encode(columns[FORM]), len(ud.words) - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except:
                raise UDError("Cannot parse HEAD '{}'".format(_encode(columns[HEAD])))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")

    return ud

def load_conllu_file(path):
    _file = open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {}))
    return load_conllu(_file)

class Score:
    def __init__(self, gold_total, system_total, correct, aligned_total=None):
        self.correct = correct
        self.gold_total = gold_total
        self.system_total = system_total
        self.aligned_total = aligned_total
        self.precision = correct / system_total if system_total else 0.0
        self.recall = correct / gold_total if gold_total else 0.0
        self.f1 = 2 * correct / (system_total + gold_total) if system_total + gold_total else 0.0
        self.aligned_accuracy = correct / aligned_total if aligned_total else aligned_total
class AlignmentWord:
    def __init__(self, gold_word, system_word):
        self.gold_word = gold_word
        self.system_word = system_word
class Alignment:
    def __init__(self, gold_words, system_words):
        self.gold_words = gold_words
        self.system_words = system_words
        self.matched_words = []
        self.matched_words_map = {}
    def append_aligned_words(self, gold_word, system_word):
        self.matched_words.append(AlignmentWord(gold_word, system_word))
        self.matched_words_map[system_word] = gold_word
class UDConfusionMatrix:
    """
    A class to represent a confusion matrix for a Universal Dependency Dataset.
	It's used to represent a confusion matrix for POS score statistic and Dependecy Relation statistic
	realized by a dependency parser.

	To generate the matrix we cycle the dataset n times, each time for each input feature to evaluate.

	Each input feature is a matrix row label and a matrix column label.
	When answer is correct we add one (+1) in the matrix row for that input feature at the column of the same input feature
	When answer by the system is wrong we add one (+1) in the matrix row for that input feature at the column of the wrong input feature
	
	In our case an input feature may be a POS, for example verb
	
		VERB PRONOUN NAME ADPOSITION ...
	VERB	40	0	0	1
	...
	PRONOUN  0 	50 	0 	3

	In this example for feature VERB the parser do well 40 times, and wrong 1, answering adposition	
    """	
    
    uDcolumns = {'XPOS':4, 'DEPREL':7}
    debug = True # adapt matrix to screen width
    debugMaxScreenWidth = 150

    def __init__(self, alignment: Alignment, metric: str, tagSet: set, cellLength : int):	
        """ 
			Parameters
			==========
			alignment: Alignment => the object used by this script to compare words  
			
			metric: str => the metric of UD is XPOS for POS and DEPREL for dependency relation
			
			tagSet: Set => if you have metric UD XPOS you have part of speech tagset ...
			
			cellLength: str => format of cell of matrix, size it base on screen width. Be aware that if you try to make cell
								to little you will get an error because you lose cypher of score/answer 

			NOTE 1:
			metric and tagset are dependent variables. Here we give them not together,
			the class must generate the set base on the metric but as this script is written
			in functional style we have already those values and we suppose they are provided together as a couple
			
			NOTE 2: 
			we tend to mantain names choosen by author of conll18_ud_eval script we use pieces 
			of their functions and algorithms to filter words and compute scores
   
        """
        self.cellLength = cellLength
        self.udConfusionMatrix = self.inputsColumn(tagSet) # initialize matrix labels
        metric = self.uDcolumns[metric]
        for inputFeature in tagSet:
            filter_fn = self.filterForGivenFeature(metric, inputFeature)
            answers = self.answersRow(tagSet)
            for words in alignment.matched_words: # words is a tuple (goldWord, systemWord)
                if filter_fn(words):
					
					# TODO : Ho ricalcato il filtro dello script che abbiamo modificato, però si perde qualcosa
					# alla riga successiva. Il filtro dice: 
					# 	se le deprel/pos di gold e syst sono le stesse => fai +1 per quella deprel
					#   se le deprel/pos di gold e syst sono diverse fai +1 per quella sbagliata

					# BUG : Per la statistica pos viene tutto corretto (evidentemente il filtro non funziona bene)
					# per la statistica deprel sembra funzionare ma nel conteggio finale si perde delle parole
					
					# puoi testare pos con il comando

					# >>> python3 conll_ud_statistics_0_9.py -stat pos -metric LEN -confusion -cell 4

					# puoi testare deprel con il comando 

					# >>> python3 conll_ud_statistics_0_9.py -stat rel -metric LAS_LEN -confusion -cell 4

					# Il filtro usato è dato dalla funzione qui sotto filterForGivenFeature(), 
					# se vedi c'è il check per il None che scarta la root e fa perder le parole

					# Ho cercato di ricalcare la tecnica di scoring della funzione

					# def alignment_score(alignment, key_fn=None, filter_fn=None, debug=False):

					# se vuoi confrontarla. Con la differenza che non passo key_fn per LAS, ma quello di UAS 
					# loro fanno una tupla di confronto e dicono solo corretto / non corretto
					# qui invece dobbiamo dire se non corretto allora quale delle non corrette

					# quello che mi sfugge è perchè, filtrata una parola, usano parent per confrontare, è li che si perde la root

					# -cell serve per dire la dimensione della cella della matrice alla print (il mio schermo è piccolino allora ho messo 4)
					# di default è 5, c'è nella classe un parametro debug per adattare la print allo schermo
					# solo la print, i dati ci sono tutti
					# puoi mettere false per stamparla tutta

					# penso di averti detto tutto
					


                    if self.gold_aligned_gold(words.gold_word.parent).columns[metric] == self.gold_aligned_system(words.system_word.parent, alignment).columns[metric]:
                        answers[words.gold_word.columns[metric]] += 1 # the system answer correctly: a point for it
                    else: 
                        answers[words.system_word.columns[metric]] += 1 # the system give wrong answer: a point for the wronged one			
            self.udConfusionMatrix[inputFeature] = answers # a row of the matrix

    def filterForGivenFeature(self, metric, inputFeature) -> callable:
        """ Filter used to parse the dataset. If feature is nsubj will be selected only words that has nsubj relation
        """
        return lambda words: words.gold_word.columns[metric] == inputFeature and words.gold_word.parent is not None and words.system_word.parent is not None # is not none to prevent root parent
    
    def inputsColumn(self, labels : set):
        """ Initialize the column of input. Every label has a corresponding row with answers for that label
			Example:

			verb   |
			--------
			pronoun|
			--------
			name   |
			-------
        """		
        inputs = {}
        for label in labels:
            inputs[label] = 0
        return inputs		

    def answersRow(self, labels : set):
        """ Initialize a row of the confusion matrix, at the moment we keep every row of
			the matrix as a map to not lose the labels related to every score
			Example:
			
					verb pronoun noun
				verb	 7	1	0
				pronoun  0	8	2
			will become:

			row1 = ("verb",{verb=7, pronoun=1, noun=0})
			row2 = ("pronoun", {verb=0, pronoun=8, noun=2})
        """
        answers = {}
        for label in labels:
            answers[label] = 0
        return answers

    def toString(self):
        matrixLength : int = self.printUDConfusionMatrixHeader()
        if self.debug: matrixLength = self.debugMaxScreenWidth
        self.printMatrixLine(matrixLength)
        for input in self.getUDConfusionMatrixInputs():
            self.printUDConfusionMatrixRow(input)
            self.printMatrixLine(matrixLength)				

    def printUDConfusionMatrixHeader(self) -> int:
        headersRow : str = self.printSpaces(self.cellLength)		
        for label in self.getUDConfusionMatrixLabels():
            headersRow +=  "|" + self.justifyLabel(label, self.cellLength + 1) + self.printSpaces(0)   
        print(headersRow if not self.debug else headersRow[0:self.debugMaxScreenWidth])
        return len(headersRow)
    
    def printUDConfusionMatrixRow(self, input : str) -> None:
        inputRow : str = self.justifyLabel(input, self.cellLength + 1)
        answersForGivenInput : dict = self.getUDConfusionMatrixAnswersForGivenInput(input)
        for answerKey in answersForGivenInput.keys():
            inputRow += "|" + self.justifyScore(answersForGivenInput[answerKey], self.cellLength + 1)
        print(inputRow  if not self.debug else inputRow[0:self.debugMaxScreenWidth])

    def getUDConfusionMatrixAnswersForGivenInput(self, input : str) -> OrderedDict:
        return OrderedDict(sorted(self.udConfusionMatrix[input].items()))		    

    def getUDConfusionMatrixLabels(self) -> list:
        return sorted(self.udConfusionMatrix.keys())		

    def getUDConfusionMatrixInputs(self) -> list:
        return sorted(self.udConfusionMatrix.keys())		    

    def	gold_aligned_gold(self, word):
        return word
    def gold_aligned_system(self, word, alignment: Alignment):
        return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None
    def justifyLabel(self, label : str, maxLength : int) -> str:
        if len(label) >= maxLength:
            return label[0:maxLength-1] 
        else:
            whiteSpacesToReachLength = maxLength - 1 - len(label)
            return label + whiteSpacesToReachLength * " "
    def justifyScore(self, score : int, maxLength : int) -> str:
        score = str(score)
        if len(score) >= maxLength:
            raise UDError("You must adjust confusion matrix cell formatting. Answer/Score: {}, maxLength/cellLength : {} => you lose cypher {}".format(score, maxLength - 1, score[maxLength-1:len(score)])) 
        else:
            whiteSpacesToReachLength = maxLength - 1 - len(score)
            return score + whiteSpacesToReachLength * " "	
    
    def printSpaces(self, numberOfSpaces:int) -> str:
        return numberOfSpaces * " "
    
    def printMatrixLine(self, matrixLength) -> None:
        print("=" * matrixLength)

    def getUDConfusionMatrixInputs(self) -> list:
        return sorted(self.udConfusionMatrix.keys())
    	
    def getTotalCount(self) -> int:
        counts : int = 0
        rows : list = self.udConfusionMatrix.values()
        for row in rows: 
            counts += sum(row.values())
        return counts
	    


def alignment_score(alignment, key_fn=None, filter_fn=None, debug=False):
    if filter_fn is not None:
        gold = sum(1 for gold in alignment.gold_words if filter_fn(gold))
        system = sum(1 for system in alignment.system_words if filter_fn(system))
        aligned = sum(1 for word in alignment.matched_words if filter_fn(word.gold_word))
    else:
        gold = len(alignment.gold_words)
        system = len(alignment.system_words)
        aligned = len(alignment.matched_words)

    if key_fn is None:
        # Return score for whole aligned words
        return Score(gold, system, aligned)

    def gold_aligned_gold(word):
        return word
    def gold_aligned_system(word):
        return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None
    correct = 0
    for words in alignment.matched_words:
        if filter_fn is None or filter_fn(words.gold_word):
			# remember, when w is root, w.parent is none
            # remap = lambda tupla : (tupla[0].columns[XPOS] if tupla[0] is not None else tupla[0], tupla[1]) 
            # print("Debug Christian: key_fn di gold ==> ", remap(key_fn(words.gold_word, gold_aligned_gold)))
            # print("Debug Christian: key_fn di syst ==> ", remap(key_fn(words.system_word, gold_aligned_system)))			
            if key_fn(words.gold_word, gold_aligned_gold) == key_fn(words.system_word, gold_aligned_system):
                correct += 1 #KEY_FN   (lambda w, ga: (ga(w.parent), w.columns[DEPREL])),
            elif debug: # 'UAS_LEN' : (lambda w, ga: ga(w.parent))
                print( key_fn(words.gold_word, gold_aligned_gold), "!=", key_fn(words.system_word, gold_aligned_system))

    return Score(gold, system, correct, aligned)

def align_words(gold_words, system_words):
    alignment = Alignment(gold_words, system_words)

    gi, si = 0, 0
    while gi < len(gold_words) and si < len(system_words):
        if gold_words[gi].is_multiword or system_words[si].is_multiword:
            # A: Multi-word tokens => align via LCS within the whole "multiword span".
            gs, ss, gi, si = find_multiword_span(gold_words, system_words, gi, si)

            if si > ss and gi > gs:
                lcs = compute_lcs(gold_words, system_words, gi, si, gs, ss)

                # Store aligned words
                s, g = 0, 0
                while g < gi - gs and s < si - ss:
                    if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                        alignment.append_aligned_words(gold_words[gs+g], system_words[ss+s])
                        g += 1
                        s += 1
                    elif lcs[g][s] == (lcs[g+1][s] if g+1 < gi-gs else 0):
                        g += 1
                    else:
                        s += 1
        else:
            # B: No multi-word token => align according to spans.
            if (gold_words[gi].span.start, gold_words[gi].span.end) == (system_words[si].span.start, system_words[si].span.end):
                alignment.append_aligned_words(gold_words[gi], system_words[si])
                gi += 1
                si += 1
            elif gold_words[gi].span.start <= system_words[si].span.start:
                gi += 1
            else:
                si += 1

    return alignment

def word_sentence_length(ud_word):
	""" Key function for max_sentence_length"""
	return ud_word.sentence_length

def word_non_projective_degree(ud_word):
	""" Key function for max_non_proj_degree"""
	return ud_word.non_projective_grade

def max_sentence_length(ds_ud_words):
	""" ritorna la lunghezza della frase più lunga del dataset ud 
	"""	
	return word_sentence_length(max(ds_ud_words, key=word_sentence_length))

def sentence_length_bins(ds_ud_words, size=10, maxsize=-1):
	""" La funzione crea una distribuzione di range di [(0*@size+1-1*@size+1),(1*@size+1- 2*@size+1),(2*@size+1,3*@size+1),...].
		 Ogni range rappresenta un bin di dimensione @size (default=10). 
		 Per esempio @size = 10 -> [range(1-11),range(11-21),range(21-31)...] = bin(1-10), bin(11-20), bin(21-30)
		 Usali per raggruppare parole in base alla distanza, per esempio word.sentence_length in bins[0] controlla se la parola
		 fa parte di una frase di lunghezza da 1 a 10.
		 Il parametro maxsize indica la lunghezza dell'ultimo range (2*@size+1,@maxsize).
		 In fase di statistica tutte le parole che appartengono a frasi di lunghezza superiore a @maxsize andranno conteggiate nel
		 bin(2*@size+1, @maxsize). 
		 Se maxsize non è specificato la funzione chiama max_sentence_length per controllare quale frase del dataset è più lunga
		 e sapere quando fermarsi.
	"""	    
	def range_to_bins(size, maxsize):
		""" La funzione prende il size di un bin e crea la distribuzione di bins tramite la funzione range
		"""
		bins = []
		for i in range(0,maxsize): # itero fino a maxsize, maxsize rappresenta il peggiore dei casi in cui bin_size = 1
			j = i + 1
			if (j*size+1 > maxsize):
				bins.append(range(i*size+1, maxsize+1))
				break
			else:
				if ((j*size+1) - (i*size+1) == size): # (i + 1)*size + 1 - i*size - 1 = i*size + size + 1 - i*size - 1 = size 
					bins.append(range(i*size+1,j*size+1))
		return bins
	# controllo che maxsize non sia < 0 altrimenti maxsize = lunghezza massima frase 
	if maxsize < 0: maxsize = max_sentence_length(ds_ud_words)
	# controllo che maxsize sia > di size
	if maxsize <= size:
		check = input("You passed a maxsize bin for sentence length <= than size? If 'Y/y' create distribution with a unique bin else exit\n")
		if check.lower() != 'y':  
			raise ValueError("Invalid maxsize={} < size={}".format(maxsize, size))
	return range_to_bins(size, maxsize)	 

			
def get_dep_distances(ds_ud_words):
	""" La funzione processa le parole del ds_ud (ds_ud.words) e ritorna un dizionario con 
		 il numero di occorrenze delle distanze tra parole dipendenti come valore e la distanza tra parole dipendenti come chiave.
		 La distanza di una dipendenza tra due parole wi, wj è data da |i - j|  
	"""
	conteggio = defaultdict(int)
	for w in ds_ud_words:
		dgold = abs(int(w.columns[ID]) - int(w.parent.columns[ID])) if w.parent != None else 0	
		if dgold != 0: conteggio[dgold] += 1
	return conteggio

def udword_distance_to_root(ud_word):
	""" La funzione calcola la distanza (int) di una parola ds_ud dalla root della frase"""
	if ud_word.parent != None: # ud_word non è root	--> posso calcolare d
		a = ud_word.parent
		while a.parent != None:
			a = a.parent # ud_word non dipende da root allora risalgo il grafo finchè $a non è root
		d = abs(int(ud_word.columns[ID]) - int(a.columns[ID]))
	else: d = 0
	return d		  

def dependency_distance(ud_word):
	return (abs(int(ud_word.columns[ID]) - int(ud_word.parent.columns[ID])) if ud_word.parent != None else 0)

def word_grade(ud_word):
	""" La funzione prende una parola ud e restituisce il numero di archi uscenti dalla dipendenza. La radice del grafo non ha dipendenze restituisco -1.
		 In teoria dei grafi è equivalente al grado del nodo. 
	"""
	#return len(ud_word.parent.functional_children) if ud_word.parent is not None else -1 
	return len(ud_word.parent.functional_children) - 1 if ud_word.parent is not None else 0 # NOTE: la correzione segue le valutazioni fatte su skype

	
def evaluation_print(metric, parametro, statistica, evaluation):
	# cambia la print in base alla statistica
		if statistica == 's': parametro = max(parametro)
		if parametro < 10: 
#			formato = "{}{}  |{:10} |{:10} |{:10} |{:10}"
#			formato2 = "{}{}  |{:10.2f} |{:10.2f} |{:10.2f} |{}" 
# commented cli print & enabled latex print
			formato = "{}{}  &{:10} &{:10} &{:10} &{:10}&{:10.2f} &{:10.2f} &{:10.2f} &{}\\\\"
		elif parametro < 100: 
#			formato = "{}{} |{:10} |{:10} |{:10} |{:10}"
#			formato2 = "{}{} |{:10.2f} |{:10.2f} |{:10.2f} |{}"
			formato = "{}{} &{:10} &{:10} &{:10} &{:10}&{:10.2f} &{:10.2f} &{:10.2f} &{}\\\\"
		else:
#			formato = "{}{}|{:10} |{:10} |{:10} |{:10}"
#			formato2 = "{}{}|{:10.2f} |{:10.2f} |{:10.2f} |{}"
			formato = "{}{}&{:10} &{:10} &{:10} &{:10}&{:10.2f} &{:10.2f} &{:10.2f} &{}\\\\"
#		print("Metric    | Correct   |      Gold | Predicted | Aligned")		
			
		print(formato.format(
			metric,
        	parametro,
        	evaluation[metric].correct,
        	evaluation[metric].gold_total,
        	evaluation[metric].system_total,
        	evaluation[metric].aligned_total or (evaluation[metric].correct if metric == "Words" else ""),
			100 * evaluation[metric].precision,
        	100 * evaluation[metric].recall,
        	100 * evaluation[metric].f1,
        	"{:10.2f}\n".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else "" 
    	))
#		print("Metric    | Precision |    Recall |  F1 Score | AligndAcc")
#		print(formato2.format(
#			metric,
#        	parametro,
#        	100 * evaluation[metric].precision,
#        	100 * evaluation[metric].recall,
#        	100 * evaluation[metric].f1,
#        	"{:10.2f}\n".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else ""))

def evaluation_header():
		print("Metric    & Correct   &      Gold & Pred & Alignd & Prec & Rec & F1 & AligndAcc \\\\")

def pos_deprel_evaluation_print(metric, parametro, statistica, evaluation):
		
#		formato = "{:7}|{}|{:10} |{:10} |{:10} |{:10}"
#		formato2 = "{:7}|{}|{:10.2f} |{:10.2f} |{:10.2f} |{}"
		formato = "{:7} & {}&{:10} &{:10} &{:10} &{:10}&{:10.2f} &{:10.2f} &{:10.2f} &{}\\\\"

		while len(parametro) != 14:
			if len(parametro) < 14:	parametro += ' '
			else: parametro -= ' '
#		print("Metric | POS o DEPREL | Correct   |      Gold | Predicted | Aligned")		
		print(formato.format(
			metric,
        	parametro,
        	evaluation[metric].correct,
        	evaluation[metric].gold_total,
        	evaluation[metric].system_total,
        	evaluation[metric].aligned_total or (evaluation[metric].correct if metric == "Words" else ""),
        	100 * evaluation[metric].precision,
        	100 * evaluation[metric].recall,
        	100 * evaluation[metric].f1,
        	"{:10.2f}\n".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else "" 
    	))
#		print("Metric | POS o DEPREL | Precision |    Recall |  F1 Score | AligndAcc")
#		print(formato2.format(
#			metric,
#        	parametro,
#        	100 * evaluation[metric].precision,
#        	100 * evaluation[metric].recall,
#        	100 * evaluation[metric].f1,
#        	"{:10.2f}\n".format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else ""))	

def pos_evaluation_header():
	print("Metric & Pos & Correct   &      Gold & Pred & Alignd & Prec &    Rec &  F1 & AligndAcc\\\\")		


def deprel_evaluation_header():
	print("Metric & DepRel & Correct   &      Gold & Pred & Alignd & Prec &    Rec &  F1 & AligndAcc\\\\")		


def precision_recall_graph(x, y1, y2, xlabel, ylabel, metrica):
	fig, ax = plt.subplots()
	ax.set_ylabel("%s (%s)" %(ylabel, metrica))
	ax.set_xlabel(xlabel)
	if not isinstance(x[0], int):  # quick fix to sentence length bug [range(1,10), 10, 20 ...]
		x[0] = max(x[0]) 
	ax.plot(x, y1, label="PRECISION")
	ax.plot(x, y2, label="RECALL" ) 
	
	ax.grid()
	ax.legend()
	plt.show()

def pos_deprel_bar_graph(x_values, y_values, y_label, x_label):
	# FIXED bug two graph to split to long featureset
	plt.bar(x_values[0:round(len(x_values)/2)], y_values[0:round(len(x_values)/2)], color="rycgb")
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	plt.show()
	plt.bar(x_values[round(len(x_values)/2):], y_values[round(len(x_values)/2):], color="rycgb")
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	plt.show()



def print_graph(s, values, precision, recall, metric, size):
	
	# The function print graphics similar to ones in Characterizing the Errors of Data-Driven Dependecy Parsing
	# Datas are generated by compute_statistics() function
	# If you need other ghraps you can add for example F1 score array inside the compute_statistics() function.
	# You can also defined reduced tagsets example: {'v','adj','cop',pron' ... } to get more human readable graphs for POS or DEP.  
		
	y_lab = "Dependency Accuracy"
	if s == 's': precision_recall_graph(values, precision, recall, "Sentence Length (bins of size {})".format(size), y_lab, metric)
	elif s == 'l': precision_recall_graph(values, precision, recall, "Dependency Length",y_lab, metric)
	elif s == 'r': precision_recall_graph(values, precision, recall, "Distance to Root",y_lab, metric)
	elif s == 'c': precision_recall_graph(values,precision,recall,"Number of Modifier Siblings",y_lab,metric)
	elif s == 'p' : precision_recall_graph(values,precision,recall,"Non-Projective Arc Degree",y_lab,metric)
	elif s == 'rel' : 
		pos_deprel_bar_graph(values, precision, "Dependency Precision", "Dependency Type (DEP)")
		pos_deprel_bar_graph(values, recall, "Dependency Recall", "Dependency Type (DEP)")
	elif s == 'pos' : 
		pos_deprel_bar_graph(values, precision, "Labelled Attachment Score (LAS) - precision", "Part Of Speech (POS)")
		pos_deprel_bar_graph(values, recall, "Labelled Attachment Score (LAS) - recall", "Part Of Speech (POS)")


def pos_dep_occurences(ds_words, dep=False, pos_t=False, min_occur=25):
	""" La funzione conteggia le occorrenze di parti del discorso di un dataset ud e
		 le occorrenze di archi di dipendenza di un dataset ud. Con parametro dep o pos
		 stampa il grafico, default dep=True. Ritorna pos e dep.
		 Il parametro min_occur screma le relazioni di dipendenza con poche occorrenze al plotting
		 del grafico per renderlo più leggibile. 
	"""
	pos = defaultdict(int)
	deprel = defaultdict(int)
	for w in ds_words:
		pos[w.columns[4]] += 1
		deprel[w.columns[7]] += 1
	if dep: print(deprel, sum(deprel.values()))
	if pos_t: print(pos, sum(pos.values()))
	#grafici
	if dep: plt.bar([i for i in deprel.keys() if deprel[i] > min_occur], [i for i in deprel.values() if i > min_occur], color = 'gbyrk')
	if pos_t: plt.bar(list(pos.keys()), pos.values(), color = 'gbyrk')
	plt.show() 
	return pos, dep

def UPOS_TAGSET(ds_ud_words):
	return {i.columns[XPOS] for i in ds_ud_words}

def DEPREL_TAGSET(ds_ud_words):
	return {i.columns[DEPREL] for i in ds_ud_words}

def clear_liste(*argv):
	for arg in argv: arg.clear()

def graph_stat_range(ds_ud_words, stat, maxsize):
	""" Function create the range for given statistics. If range is not specified, i search for the max value for the given statistics
		 in the ud_dataset.
	"""
	if maxsize != -1: stat_range = range(0, maxsize+1) # se specificato un maxsize creo il range
	# se il range non è specificato chiamo una funzione che in base alla statistica cerca:
	# max_dependency_length, max_distance_to_root, max_siblings, max_non_proj_degree
	else:
		switcher = {
		'l' : dependency_distance,
		'r' : udword_distance_to_root,
		'c' : word_grade,
		'p' : word_non_projective_degree,
		}
		stat_range = range(0, switcher[stat](max(ds_ud_words, key=switcher[stat]))+1)
	return stat_range
	
def process_statistic_arguments(stat, size, maxsize, metric, ds_words, confusion):
	""" The function process parameters passed by command line and set filter, metric and iterator to launch the statistics """
	RANGE_ON = False
	WORD_ON = False
	MAX_RANGE_ON = False
	CONFUSION = False 
	matrixStat = ""
	if stat == 's':
		if size == -1: 
			size = int(input("ATTENTION: you didn't specify the bin size for {} statistics. Input size > 0 to continue\n".format(stat)))
			if not (size > 0): raise UDError("Invalid size for sentence length statistics")    
		iterator = sentence_length_bins(ds_words, size, maxsize)
		print_header = evaluation_header
		print_function = evaluation_print
		RANGE_ON = True
		MAX_RANGE_ON = True
	elif stat in {'l','r','c','p'}:
		iterator = graph_stat_range(ds_words, stat, maxsize)
		print_header = evaluation_header
		print_function = evaluation_print
		MAX_RANGE_ON = True
	elif stat == 'pos':
		WORD_ON = True
		print_header = pos_evaluation_header
		print_function = pos_deprel_evaluation_print
		iterator = UPOS_TAGSET(ds_words)
		matrixStat = 'XPOS'
		CONFUSION = True
	elif stat == 'rel':
		WORD_ON = True
		iterator = DEPREL_TAGSET(ds_words)
		print_header = deprel_evaluation_header
		print_function = pos_deprel_evaluation_print
		CONFUSION = True
		matrixStat = 'DEPREL'
	if (not CONFUSION) and (confusion):
		raise UDError("Confusion matrix can be generated only for part of speech and dependency relation statistics")
	if (WORD_ON) and (metric != 'LAS-len'):
		raise UDError("For lexical statistics 'pos' or 'r' you need to set LAS_LEN metric. You passed {} metric".format(metric))		
	if (not RANGE_ON) and (size != -1): print("ATTENTION: range argument is unused, you don't need to specify it for {} statistics\n".format(stat))
	if (not MAX_RANGE_ON) and (maxsize != -1): print("ATTENTION: maxrange argument is unused, you don't need to specify it for {} statistics\n".format(stat))  
	return iterator, print_function, matrixStat, print_header


def compute_statistics(argument_filter, argument_metric, iterator, print_function, ds_ud, alignment, print_header):
	# FILTER_MAP[i][1] is special_filter for last condition, it grouped all the words not counted 
	FILTER_MAP = { 's' : ((lambda w: w.sentence_length in i),(lambda w: (w.sentence_length in i) or (w.sentence_length > max(i)))), 
				 	'l' : ((lambda w: dependency_distance(w) == i),(lambda w: dependency_distance(w) >= i)),  
					'r' : ((lambda w: udword_distance_to_root(w) == i),(lambda w: udword_distance_to_root(w) >= i)),
					'c' : ((lambda w: word_grade(w) == i),(lambda w: word_grade(w) >= i)),
					'p' : ((lambda w: w.non_projective_grade == i),(lambda w: w.non_projective_grade >= i)),
				 'pos' : ((lambda w: w.columns[XPOS] == i),(lambda w: w.columns[XPOS] == i)),
				 'rel' : ((lambda w: w.columns[DEPREL] == i),(lambda w: w.columns[DEPREL] == i)), 	}	

	METRIC_MAP = { 
				'LAS-len' : (lambda w, ga: (ga(w.parent), w.columns[DEPREL])),
				'UAS-len' : (lambda w, ga: ga(w.parent))}

	stat_filter, last_stat_filter = FILTER_MAP[argument_filter]
	
	gold_total, sys_total, VALUES, PRECISION, RECALL = [],[],[],[],[]

	# The iterable object is transformed into an iterator to handle as a special condition last iteration 
	special_iterator = iter(iterator)
	i = next(special_iterator) # First value
	VALUES.append(i)
	print_header()
	for j in special_iterator:	
		# compute statistics with i
		evaluation = { argument_metric : alignment_score(alignment, key_fn=METRIC_MAP[argument_metric], filter_fn=stat_filter) }
		print_function(argument_metric, i, argument_filter, evaluation)
		i = j # Set j as last value
		gold_total.append(evaluation[argument_metric].gold_total)
		sys_total.append(evaluation[argument_metric].system_total)
		VALUES.append(i if argument_filter != "s" else max(i))
		PRECISION.append(evaluation[argument_metric].precision)
		RECALL.append(evaluation[argument_metric].recall)
	# out of the cycle you call the filter for last iteration
	evaluation = {argument_metric : alignment_score(alignment, key_fn=METRIC_MAP[argument_metric], filter_fn=last_stat_filter)} # filtro speciale per ultima condition
	print_function(argument_metric, i, argument_filter, evaluation)
	# check words counted of system and gold ds are the same
	gold_total.append(evaluation[argument_metric].gold_total)
	sys_total.append(evaluation[argument_metric].system_total)
	#VALUES.append(i if argument_filter != "s" else max(i))
	PRECISION.append(evaluation[argument_metric].precision)
	RECALL.append(evaluation[argument_metric].recall)
	if sum(gold_total) != sum(sys_total): raise UDError("Total number of gold words evaluated {} is different from system one {} for statistic {}".format(sum(gold_total), sum(sys_total), argument_filter))  	
	assert sum(gold_total) == len(ds_ud.words)  # check all the words has been counted 
	return VALUES, PRECISION, RECALL, gold_total

def main():
	STATISTICS = { 's','l','r','c','p','pos','rel'}
	METRICS = { 'LAS-len', 'UAS-len' } 
	
	# Statistics help
	stat_help = """ -stat [statistics_to_compute]                                                                           
						 every shortcut correspond to a statistics from the paper 

						 *Characterizing the Errors of Data-Driven Dependency Parsing Models*
						                                                                                                                                                                                 
						 s => sentence length |
						 l => dependency length |
						 r => distance to root |
						 c => number of modifier siblings |
						 p => non projective arc degree |
						 pos => part of speech |
						 rel => dependency type
					"""
	max_range_help = """ Max distance of the dependency between two words, l | 
						  		Max number of modifier siblings of a word, c |
						  		Max grade of non projectivity, p |
						  		Max size of a bin for sentence length, s.
						  		By default the statistics will be compute overall dataset.
						  		If specified all the values greater than this will be collect together.
						  		You should fit it to every statistics.
						  		
						  		Example:  
					 	  """	 

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-stat", type=str, choices=STATISTICS, required=True, help=stat_help)
	parser.add_argument("-metric", type=str, choices=METRICS, required=True, help="UAS = unlabelled attachment score, LAS = labelled attachment score")
	parser.add_argument("-range", type=int, default=-1, help="Size of a bin for sentence length statististic")
	parser.add_argument("-maxrange", type=int, default=-1, help=max_range_help)
	parser.add_argument("-graph", default=False, action="store_true", help="Pass this to view graphics")
	parser.add_argument("-confusion", default=False, action="store_true", help="Pass this to view confusion matrix, can be used with part of speech or dependency relation statistics")
	parser.add_argument("-cell", type=int, default=5, help="The size of a confusion matrix cell")

	parser.add_argument("-gold", default="gold.conllu", help="Gold dataset path")
	parser.add_argument("-pred", default="syst.conllu", help="Predicted dataset path")


	args = parser.parse_args()

	# Load datasets
	
	gold_ud = load_conllu_file(args.gold)
	system_ud = load_conllu_file(args.pred)
	alignment = align_words(gold_ud.words, system_ud.words)
	
	# Join all the words to calculate max values and retrieve all the tags (if some little values are missing from one dataset)
	big_ds = gold_ud.words + system_ud.words # you can change the code passing iteratively just the references to the two dataset inside functions (computing dinamically and saving RAM)
	 
		
	# pass argument to function to set statistics dependent iterator and evaluation parameters	
	
	try:
		iterator, print_function, matrixStat, print_header = process_statistic_arguments(args.stat, args.range, args.maxrange, args.metric, big_ds, args.confusion)
	except UDError as e:
		print("UDError: {}".format(e))
	try:
		VALUES, PRECISION, RECALL, gold_total = compute_statistics(args.stat, args.metric, iterator, print_function, gold_ud, alignment, print_header)
	except UDError as e:
		print("UDError: {}".format(e))
	
	# print graphs
	if args.graph:
		print_graph(args.stat, VALUES,PRECISION,RECALL,args.metric, args.range)
	
	# print confusion matrix for dependency relations or part of speech statistics
	if args.confusion:
		confusionMatrixUD = UDConfusionMatrix(alignment, matrixStat, iterator, args.cell)
		try: 
			confusionMatrixUD.toString()
		except UDError as e:
			print("UDError: {}".format(e))

		# TODO : remove this print when debugged
		print("""===============\nTest:\n\ngold words: {} | matrix counted words: {}\n==========================
			  """.format(len(gold_ud.words), confusionMatrixUD.getTotalCount()))


if __name__ == '__main__':
	main()

# test that can be executed with 'python -m unittest conll_ud_statisics.py'
# run it with -b to avoid print stuff
	
class TestConlluStatsCount(unittest.TestCase):
	@staticmethod
	def _load_words(words):
		"""Prepare fake CoNLL-U files with fake HEAD to prevent multiple roots errors."""
		lines, num_words = [], 0
		for w in words:
			parts = w.split(" ")
			if len(parts) == 1:
				num_words += 1
				lines.append("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_".format(num_words, parts[0], int(num_words>1)))
			else:
				lines.append("{}-{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_".format(num_words + 1, num_words + len(parts) - 1, parts[0]))
				for part in parts[1:]:
					num_words += 1
					lines.append("{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_".format(num_words, part, int(num_words>1)))
		return load_conllu((io.StringIO if sys.version_info >= (3, 0) else io.BytesIO)("\n".join(lines+["\n"])))	

	def _test_compute_statistics(self, stat, size, maxsize, metric, d1, d2):	
		gold = self._load_words(d1)
		system = self._load_words(d2)
		alignment = align_words(gold.words, system.words)
		big_ds = gold.words + system.words
		iterator, print_function = process_statistic_arguments(stat, size, maxsize, metric, big_ds)
		l1,l2,l3,gold_total = compute_statistics(stat, metric, iterator, print_function, gold, alignment)
		self.assertEqual(sum(gold_total), len(gold.words))

	def test_compute_statistics(self):
		""" Test creates two syntetic datasets and test all the statistics, all the metrics with multiple parameters of range and maxrange.
			 The creation of dataset used the function defined in conll18_ud_eval.py test class.
		"""
		STATISTICS = { 's','l','r','c','p','pos','rel'}
		METRICS = { 'LAS-len', 'UAS-len' }
		sizes = [-1,0,1,3,5,10] 
		maxsizes = [-1,11,22,25,30,999] # note that maxsize is always greater than size, in main there's a check on this condition
		file1 = [i for i in 'rrascarwefwdkcjwsxdbqdxwuidzwdzkhw']
		file2 = [i for i in 'asdnsdnkaSDXUWHQDKQWDKHWQDKAKksdjn']
		for metric in METRICS:
			for stat in STATISTICS:
				for size in sizes:
					for maxsize in maxsizes:
						if stat in {'pos','rel'} and metric == 'UAS-len': break # with pos and rel only LAS is admitted 
						if stat == 's' and size < 1: break # there are no sentences of length 0 
						if stat == 'c': break # TEST non funziona con c. Nel main lavora bene. Forse è dovuto ad una semplificazione del dataset sintetico 
						self._test_compute_statistics(stat,size,maxsize,metric,file1,file2)




# Note:
	# aggiunti attributi per le statistiche al caricamento delle parole
	# semplificato il main() che chiama due funzioni -> una per la gestione degli argomenti a linea di comando: process_statistic_argumnets(); 
	#    una per il calcolo delle statistiche compute_statistics();
	# tutte le parole del dataset vengono sempre conteggiate per qualsiasi statistica
	# nel caso in cui sia specificato un parametro max_value le rimanenze vengono raggruppate nell'ultima iterazione di compute_statistics()  
	# I grafici vengono stampati seguendo il paper. Per print diverse (F1 score o -human readable) va modificato il codice ->
	# due suggerimenti: aggiungere array F1 in compute statistics, ridurre i tagset di 'pos' o 'rel'.      

	# il programa non supporta multi_word_datasets. Potrebbero uscire bugs  
	
	
