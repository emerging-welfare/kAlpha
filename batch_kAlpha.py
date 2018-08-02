from __future__ import print_function
import sys
import re
import pandas as pd
import argparse
import itertools
from pynlpl.formats import fql, folia
from glob import glob

all_result = []

global disaagrements_df

disaagrements_df = pd.DataFrame(columns=["tag","annot1","annot2","sentence"])

def sorted_nicely(l):
    # Copied from this post
    # https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    # Thank you Daniel
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def sorted_nicely2(l):
    # Copied from this post (Added a little tweak at the end for it to work on list of lists)
    # https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    # Thank you Daniel
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key[0])]
    return sorted(l, key=alphanum_key)


def getData(entity, annot, sp, sent_len, df):

    wordList = []
    for wref in entity.wrefs():
        wordList.append(re.sub(r'.*(s\.\d+\.w\.\d+)$', r'\g<1>', wref.id))

    wordList = sorted_nicely(wordList)
    # Check if there is any space in entity. Divide them if there is.
    i = 0
    current_sent = int(re.search(r'^s\.(\d+)', wordList[0]).group(1))

    while i < len(wordList) - 1 and len(wordList) > 1:
        sent = int(re.search(r's\.(\d+)\.w\.\d+$', wordList[i]).group(1))
        next_sent = int(re.search(r's\.(\d+)\.w\.\d+$',
                                  wordList[i + 1]).group(1))
        word = int(re.search(r'(\d+)$', wordList[i]).group(1))
        next_word = int(re.search(r'(\d+)$', wordList[i + 1]).group(1))

        if word + 1 != next_word or (next_word == 1 and next_sent == sent + 1):
            first_word = int(re.search(r'(\d+)$', wordList[0]).group(1))
            first_sent = int(
                re.search(r's\.(\d+)\.w\.\d+$', wordList[0]).group(1))
            first_word += sp
            word += sp

            if first_sent == current_sent + 1:
                first_word += sent_len

            if sent == current_sent + 1:
                word += sent_len

            df = df.append({"entity": [str(first_word) + ":" + str(word), entity.cls],
                            "annotator": annot, "focus": entity.set, "tag": entity.cls}, ignore_index=True)
            for j in range(0, i + 1):
                del wordList[0]

            i = 0
            if len(wordList) == 1:
                next_word += sp
                if next_sent == current_sent + 1:
                    next_word += sent_len

                df = df.append({"entity": [str(next_word) + ":" + str(next_word), entity.cls],
                                "annotator": annot, "focus": entity.set, "tag": entity.cls}, ignore_index=True)
                del wordList[0]

        else:
            i = i + 1

    if len(wordList) > 0:
        first_word = int(re.search(r'(\d+)$', wordList[0]).group(1))
        first_sent = int(re.search(r's\.(\d+)\.w\.\d+$', wordList[0]).group(1))
        last_word = int(re.search(r'(\d+)$', wordList[-1]).group(1))
        last_sent = int(re.search(r's\.(\d+)\.w\.\d+$', wordList[-1]).group(1))
        first_word += sp
        last_word += sp
        if first_sent == current_sent + 1:
            first_word += sent_len

        if last_sent == current_sent + 1:
            last_word += sent_len

        df = df.append({"entity": [str(first_word) + ":" + str(last_word), entity.cls],
                        "annotator": annot, "focus": entity.set, "tag": entity.cls}, ignore_index=True)

    return df


def organize_data(entities, doc_length, args):

    out_list = []
    start_point = 1
    if len(entities) == 0:
        out_list.append([1, doc_length, args.empty])
        return out_list

    for entity in entities:
        # ESP = Entity Start Point , EEP = Entity End Point
        ESP = int(re.search(r'^(\d+):', entity[0]).group(1))
        EEP = int(re.search(r':(\d+)$', entity[0]).group(1))
        if ESP > start_point:
            out_list.append([start_point, ESP - 1, args.empty])

        elif ESP < start_point:
            print("Duplicates or Overlapping Tags")
            continue

        out_list.append([ESP, EEP, entity[1]])
        start_point = EEP + 1

    last_point = out_list[-1][1]
    if last_point < doc_length:
        out_list.append([last_point + 1, doc_length, args.empty])

    return out_list


def getUnion(a, b):
    return max(a[1], b[1]) - min(a[0], b[0]) + 1


def getIntersect(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) + 1


def getLength(a):
    return a[1] - a[0] + 1


def encapsulates(a, b):
    if getIntersect(a, b) == getLength(b):
        return True

    return False


def getMetric(a, b):
    if a[2] == b[2]:
        return 0

    return 1

def all_docs(kalpha_list,tags):
    result = {}
    for tag in list(set(tags)):
        result[tag] = {}
        result[tag]['count'] = 0
        result[tag]['kalpha'] = 0.0
    for kalpha in kalpha_list:
        for tag in kalpha.keys():
            result[tag]['count'] += kalpha[tag]['weight']
            result[tag]['kalpha'] += kalpha[tag]['kalpha']*kalpha[tag]['weight']
    for tag in result.keys():
        if result[tag]['count'] != 0:
            result[tag]['kalpha'] = result[tag]['kalpha'] / result[tag]['count']
    return result

def getid(a,ind,sentence_lengths,sentence_ids):

    for i,sent_len in enumerate(sentence_lengths):
        if a[0] <= sent_len:
            #print(a[0]+ind)
            s = sentence_ids[i]
            if i == 0:
                id = s + ".w." + str(a[0] + ind)
            else:
                id = s + ".w." + str(a[0] + ind - sentence_lengths[i-1])
            #print(id)
            return id,s

def addwords(a,b,tag,filename,sentence_lengths,args):

    global disaagrements_df
    doc = folia.Document(file=args.document[0]+filename)

    sentence_ids = []
    for paragraph in doc.paragraphs():
        for sentence in paragraph.sentences():
            sentence_ids.append(sentence.id)

    annot1 = ""
    annot2 = ""

    filename = re.sub(r"\.folia\.xml$", r"", filename)

    if a[2] != args.empty:
        for ind in range(0,a[1]-a[0]+1):
            asd,sentence2 = getid(a,ind,sentence_lengths,sentence_ids)
            #print(asd + "    annot1")
            if annot1 == "":
                annot1 = doc[asd].text()
            else:
                annot1 = annot1 + " " + doc[asd].text()

    if b[2] != args.empty:
        for ind in range(0,b[1]-b[0]+1):
            asd,sentence2 = getid(b,ind,sentence_lengths,sentence_ids)
            #print(asd + "    annot2")
            if annot2 == "":
                annot2 = doc[asd].text()
            else:
                annot2 = annot2 + " " + doc[asd].text()
    #print(annot1)
    #print(annot2)
    #print(sentence2)
    sentence2 = doc[sentence2]
    #print("hey")
    disaagrements_df = disaagrements_df.append({"tag":tag,"annot1":annot1,"annot2":annot2,"sentence":sentence2},ignore_index=True)

    return

def calculate_Kalpha(in_df, annots, args, sentence_lengths, filename):
    pairs = list(itertools.combinations(annots, 2))
    observed_nom = 0
    observed_denom = 0
    expected_nom = 0
    expected_denom = 0
    empty_count = 0
    if in_df.empty:
        return {"kalpha":0.0,"weight":observed_denom}


    for pair in pairs:
        entities1 = in_df[in_df.annotator == pair[0]].entities.tolist()[0]
        entities2 = in_df[in_df.annotator == pair[1]].entities.tolist()[0]
        print("Entities1 : " + str(entities1))
        print("Entities2 : " + str(entities2))
        if entities1 == entities2:
            #print(entities1), 291, 'empty'], [292,
            for g in entities1:
                if g[2] != args.empty:
                    observed_denom += 1

            continue

        for g in entities1:
            for h in entities2:
                intersect = getIntersect(g, h)
                if intersect > 0:
                    if g[2] != args.empty and h[2] != args.empty:
                        observed_nom += getUnion(g, h) - intersect * (1 - getMetric(g, h))
                        observed_denom += 1

                        if getMetric(g,h) == 0:
                            tag2 = g[2]
                            addwords(g,h,tag2,filename,sentence_lengths,args)

                    elif g[2] != args.empty and h[2] == args.empty and encapsulates(h, g):
                        observed_nom += 2 * getLength(g)
                        observed_denom += 1
                        tag2 = g[2]
                        addwords(g,h,tag2,filename,sentence_lengths,args)

                    elif g[2] == args.empty and h[2] != args.empty and encapsulates(g, h):
                        observed_nom += 2 * getLength(h)
                        observed_denom += 1
                        tag2 = h[2]
                        addwords(g,h,tag2,filename,sentence_lengths,args)

                    elif g[2] == args.empty and h[2] == args.empty:
                        empty_count += intersect

    if observed_nom == 0:
        return {"kalpha":1.0,"weight":observed_denom}

    observed = float(observed_nom / observed_denom)
    entities = [i for x in in_df.entities.tolist()
                for i in x if i[2] != args.empty]
    expected = 0.0
    for g in entities:
        seenItself = False
        for h in entities:
            if g == h and not seenItself:
                seenItself = True
                continue

            leng = getLength(g)
            lenh = getLength(h)
            expected_nom += leng * leng + lenh * lenh + leng * lenh * getMetric(g, h)
            expected_denom += leng + lenh

    if expected_denom != 0:
        expected = float(expected_nom / expected_denom)

    if expected == 0:
        return {"kalpha":0.0,"weight":observed_denom}

    kalpha = 1.0 - observed / expected
    return {"kalpha":kalpha,"weight":observed_denom}


def getResult(all_df, annots, args, doc_length, sentence_lengths, filename):

    for i,sent_len in enumerate(sentence_lengths):
        if i != 0:
            sentence_lengths[i] = sentence_lengths[i-1] + sent_len

    #print(sentence_lengths)

    if not args.overlap:
        entities_df = pd.DataFrame()
        for annot in annots:
            entities = all_df[all_df.annotator == annot].entity.tolist()
            entities = sorted_nicely2(entities)
            entities = organize_data(entities, doc_length, args)
            entities_df = entities_df.append(
                {"annotator": annot, "entities": entities}, ignore_index=True)

        return calculate_Kalpha(entities_df, annots, args, sentence_lengths, filename)

    else:
        tags = all_df.tag.unique()
        kAlpha_tags = {}
        for tag in tags:
            entities_df = pd.DataFrame()
            for annot in annots:
                entities = all_df[(all_df.annotator == annot) & (
                    all_df.tag == tag)].entity.tolist()
                entities = sorted_nicely2(entities)
                entities = organize_data(entities, doc_length, args)
                entities_df = entities_df.append(
                    {"annotator": annot, "entities": entities}, ignore_index=True)

            kAlpha_tags[tag] = calculate_Kalpha(entities_df, annots, args, sentence_lengths, filename)

        return kAlpha_tags


def main():
    parser = argparse.ArgumentParser(description="This is a script for calculating Krippendorff's Alpha between two or more annotators. You can find the latest paper on the metric in this link : http://web.asc.upenn.edu/usr/krippendorff/m-Replacement%20of%20section%2012.4%20on%20unitizing%20continua%20in%20CA,%203rd%20ed.pdf (Please give document(s) first, then pass options.)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--empty', type=str, help="If you have a category named empty pass this argument with a string.",
                        default="empty", action='store', required=False)
    parser.add_argument('-F', '--Folia', help="If your inputs are folia documents",
                        action='store_true', default="", required=False)
    parser.add_argument('-s', '--set', type=str,
                        help="Set definition (This is for Folia docs. Required if there is more than one set in the document)", action='store', required=False)
    parser.add_argument('-o', '--overlap', help="If your categories(tags) are overlapping",
                        action='store_true', default="", required=False)
    # This assumes that one tag doesn't apply to same word more than once
    # For the csv file's format see https://github.com/emerging-welfare/kAlpha/example.csv
    parser.add_argument('document', nargs='+',
                        help="CSV File For the csv file's format see https://github.com/OsmanMutlu/kAlpha/example.csv (You can choose multiple Folia docs too.)")
    args = parser.parse_args()

    if args.Folia:

        all_tags = []
        files = glob(args.document[0] + "http*")
        for filename in files:

            filename = re.sub(r"^.*\/([^\/]*)$", r"\g<1>", filename)
            print(filename)
            sentence_lengths = []
            all_df = pd.DataFrame()
            # All docs' sentences should be same length.
            for i, docfile in enumerate(args.document):
                docfile = docfile + filename
                doc = folia.Document(file=docfile)
                if i == 0:
                    for j, sentence in enumerate(doc.sentences()):
                        sentence_lengths.append(len(sentence))

                start_point = 0
                for h, sentence in enumerate(doc.sentences()):
                    for layer in sentence.select(folia.EntitiesLayer):
                        for entity in layer.select(folia.Entity):
                            all_df = getData(
                                entity, "annot" + str(i + 1), start_point, sentence_lengths[h], all_df)

                    start_point = start_point + sentence_lengths[h]

            doc_length = sum(sentence_lengths)

            if len(all_df.index) == 0:
                continue

            if args.set:
                all_df = all_df[all_df.focus == args.set]

            annotators = all_df.annotator.unique()
            asd_tags = all_df.tag.unique()
            all_tags.extend(asd_tags)

            result = getResult(all_df, annotators, args, doc_length, sentence_lengths, filename)
            all_result.append(result)

    else:
        if len(args.document) != 1:
            sys.exit("Please give only one document for CSV input.")

        all_df = pd.read_csv(args.document[0], header=None, names=[
                             'annotator', 'start', 'end', 'tag'], comment="#")
        with open(args.document[0], "r") as f:
            doc_length = f.readline().strip()

        doc_length = re.search(r'length=(\d+)', doc_length)
        if doc_length:
            doc_length = int(doc_length.group(1))

        else:
            print("You didn't write the documents length in the first line of csv.")
            # This is arbitrary.
            doc_length = 200000

        all_df['entity'] = all_df[['start', 'end', 'tag']].apply(
            lambda x: [str(x[0]) + ":" + str(x[1]), str(x[2])], axis=1)
        annotators = all_df.annotator.unique()
    # We are finished with getting data. Both folia's and csv's data layout is same
    print(all_docs(all_result, all_tags))

if __name__ == "__main__":
    main()
    writer = pd.ExcelWriter('disaagrements.xlsx')
    disaagrements_df.to_excel(writer)
    writer.save()
