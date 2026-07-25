import random


def makeDataFormatter(nlp, tokenizer):
    """
        Creates the complete preprocessing pipeline used by Dataset.map().

        For every document the pipeline:
        1. Finds a valid noun-pronoun coreference pair.
        2. Flattens the document into a continuous token sequence.
        3. Selects two distractor noun candidates.
        4. Extracts a local context span.
        5. Removes ambiguous or unsuitable examples.
        6. Randomises candidate order to remove positional bias.
        7. Encodes the example into the SpanBERT multiple-choice format.

        Invalid examples return None values and are removed later by the
        dataset wrapper.
    """
    def formatData(data):
        noun, nounMention, pronoun, pronounMention, cluster = extarctNounPronoun(data["sentences"], data["coref_chains"], nlp)

        if noun == None or pronoun == None:
            return {"input_ids": None, "attention_masks": None, "label": None, "span": None, "candidates": None, "pronoun": None}

        flatSent, flatCluster, nounID, pronounID = flattenUpTo(data["sentences"], nounMention, pronounMention, nlp, cluster)

        extraNounOne, extraNounOneID, extraNounTwo, extraNounTwoID = extractNouns(flatSent, nounID, pronounID, flatCluster)
        
        # This catches cases that may make the choice ambiguous.
        if extraNounOne == None or (noun == extraNounOne) or (noun == extraNounTwo) or (extraNounOne == extraNounTwo) or (nounID >= pronounID):
            return {"input_ids": None, "attention_masks": None, "label": None, "span": None, "candidates": None, "pronoun": None}

        span = extractSpan(flatSent, flatCluster, nounID, pronounID, extraNounOneID, extraNounTwoID)


        if span == None:
            return {"input_ids": None, "attention_masks": None, "label": None, "span": None, "candidates": None, "pronoun": None}


        # Shuffle candidates to all true candidates being label 0.
        candidates = [noun, extraNounOne, extraNounTwo]
        random.shuffle(candidates)
        for i, cand in enumerate(candidates):
            if cand == noun:
                label = i
                break
            
        input_ids, attention_maks = getEncodedInputs(tokenizer, span, candidates, pronoun)
        
        return {"input_ids": input_ids, "attention_masks": attention_maks, "label": label, "span": span, "candidates": candidates, "pronoun": pronoun}
    return formatData


def extarctNounPronoun(sentences, coref_chains, nlp):
    """
        Searches the annotated coreference chains for a usable noun-pronoun pair.

        Only single-token mentions are considered in order to simplify the multiple-choice formulation.
        Once both a noun antecedent and a referring pronoun have been found within the same coreference chain,
        the corresponding mentions are returned.
    """
    for chain in coref_chains:
        noun = None
        nounMention = None 
        pronoun = None
        pronounMention = None
        for ment in chain:
            if ment[1] != ment[2]:
                continue
            doc = nlp(sentences[ment[0]]["tokens"][ment[1]]["text"])
            if doc[0].pos_ == "PROPN" or doc[0].pos_ == "NOUN":
                noun = sentences[ment[0]]["tokens"][ment[1]]["text"]
                nounMention = ment
            elif doc[0].pos_ == "PRON":
                pronoun = sentences[ment[0]]["tokens"][ment[1]]["text"]
                pronounMention = ment

            if noun != None and pronoun != None:
                return noun, nounMention, pronoun, pronounMention, chain
    return noun, nounMention, pronoun, pronounMention, None

    

def flattenUpTo(sentences, nounMention, pronounMention, nlp, cluster):
    """
        Converts the PreCo document representation into a single flattened token sequence.

        The original PreCo dataset stores mentions as sentence-relative token indices.
        Flattening produces one ontinuous sequence while updating mention indices so they remain valid after concatenation.

        Only sentences up to the pronoun are retained because later context cannot influence antecedent selection.
    """
    mention = max(nounMention, pronounMention)
    topSentenceID = mention[0]
    flatSentences = []
    flatClusters = []
    nounID = nounMention[1]
    pronounID = pronounMention[1]
    id = 1
    idUpToSent = 1
    for sent in sentences:
        if sent["id"] == topSentenceID+2:
            break
        
        clustInSent = [clust for clust in cluster if clust[0]+1 == sent["id"]]
        
        idUpToSent = id

        for clust in clustInSent:
            flatClusters.append(clust[1]+id)
        if sent["id"] == nounMention[0]+1:
            nounID = nounMention[1] + idUpToSent
        if sent["id"] == pronounMention[0]+1:
            pronounID = pronounMention[1] + idUpToSent

        for tok in sent["tokens"]:
            doc = nlp(tok["text"])
            if doc[0].pos_ == "NOUN" or doc[0].pos_ == "PROPN":
                flatSentences.append({"id": id, "text": tok["text"], "noun": 1})
            else:
                flatSentences.append({"id": id, "text": tok["text"], "noun": 0})
            id += 1

    return flatSentences, flatClusters, nounID, pronounID




def extractNouns(flatSentence, nounID, pronounID, cluster):
    """
        Selects two distractor noun candidates.

        Candidate antecedents are restricted to nouns occurring before the
        pronoun and outside the gold coreference chain. This creates a
        balanced three-choice classification problem while avoiding obvious
        duplicate or correct answers.
    """
    extraNounOne = None
    extraNounTwo = None
    for tok in flatSentence[::-1]:
        if tok["id"] >= pronounID:
            continue
        elif tok["id"] == nounID:
            continue
        elif tok["noun"] == 1 and tok["id"] not in cluster:
            if extraNounOne == None:
                extraNounOne = tok
                
            elif extraNounOne["text"] != tok["text"]:
                extraNounTwo = tok

        if extraNounTwo is not None:
            return extraNounOne["text"], extraNounOne["id"], extraNounTwo["text"], extraNounTwo["id"]
    return None, None, None, None





def extractSpan(flatSentence, clusterIDs, nounID, pronounID, extraNounOneID, extraNounTwoID):
    """
        Extracts the local context span used by the model.

        A small window surrounding the pronoun and all candidate nouns is selected.
        Examples are rejected if additional mentions from the original coreference chain appear
        inside the span, as these could introduce ambiguity into the constructed classification problem.
    """
    bottomIDX = max(min(nounID, pronounID, extraNounOneID, extraNounTwoID) - 3, 0)
    topIDX = min(max(nounID, pronounID, extraNounOneID, extraNounTwoID) + 2, len(flatSentence) - 1)

    clusterIDs.remove(nounID)
    clusterIDs.remove(pronounID)
    
    if any(clusterIDs) >= bottomIDX and any(clusterIDs) <= topIDX:
        return None
    

    span = " ".join([text["text"] for text in flatSentence[bottomIDX:topIDX]])

    return span



def getEncodedInputs(tokenizer, span, candidates, pronoun):
    """
        Converts each candidate antecedent into an independent SpanBERT multiple-choice input.

        Each candidate is paired with the same context span while only the proposed antecedent changes.
    """
    encodedPronoun = [tokenizer(span, f"{pronoun} refers to {noun}", padding = False, truncation = True) for noun in candidates]
    input_ids = []
    attention_mask = []

    for enc in encodedPronoun:
        input_ids.append(enc["input_ids"])
        attention_mask.append(enc["attention_mask"])

    return input_ids, attention_mask

