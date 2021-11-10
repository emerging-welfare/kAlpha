from __future__ import print_function
import sys
import re
import pandas as pd
import argparse
import itertools

# NOTE: You can comment next line if you aren't working with folia documents.
from pynlpl.formats import folia

def sorted_nicely(l): # alphanumeric string sort
    # Copied from this post -> https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key[0])]
    return sorted(l, key=alphanum_key)

def organize_data(entities, doc_length, empty_tag):
    """
    Organize the annotations in an acceptable format for calculate_kalpha function.
    """

    out_list = []
    start_point = 0
    if len(entities) == 0:
        out_list.append([1, doc_length, empty_tag])
        return out_list

    for entity in entities:
        # ESP = Entity Start Point , EEP = Entity End Point
        ESP = entity[0]
        EEP = entity[1]
        if ESP > start_point:
            out_list.append([start_point, ESP - 1, empty_tag])

        elif ESP < start_point:
            # TODO: Too many of this error, investigate!
            print("Duplicates or Overlapping Tags")
            continue

        out_list.append([ESP, EEP, entity[-1]])
        start_point = EEP + 1

    last_point = out_list[-1][1]
    if last_point < doc_length:
        out_list.append([last_point + 1, doc_length, empty_tag])

    return out_list


def get_union(a, b):
    return max(a[1], b[1]) - min(a[0], b[0]) + 1

def get_intersect(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) + 1

def get_length(a):
    return a[1] - a[0] + 1

def encapsulates(a, b):
    if get_intersect(a, b) == get_length(b):
        return True

    return False

def get_metric(a, b):
    if a[2] == b[2]:
        return 0

    return 1

def calculate_kalpha(in_entities, annots, empty_tag):
    pairs = list(itertools.combinations(list(range(len(annots))), 2))
    observed_nom = 0
    observed_denom = 0
    expected_nom = 0
    expected_denom = 0
    empty_count = 0
    weight = 0
    if len(in_entities) == 0:
        return 0.0, 0

    for pair in pairs:
        entities1 = in_entities[pair[0]]
        entities2 = in_entities[pair[1]]
        if entities1 == entities2:
            for g in entities1:
                if g[2] != empty_tag:
                    observed_denom += 1
                    weight += 2 * get_length(g)

            continue

        for g in entities1:
            for h in entities2:
                intersect = get_intersect(g, h)
                if intersect > 0:
                    if g[2] != empty_tag and h[2] != empty_tag:
                        observed_nom += get_union(g, h) - intersect * (1 - get_metric(g, h))
                        observed_denom += 1
                        weight += get_length(g) + get_length(h)

                    elif g[2] != empty_tag and h[2] == empty_tag and encapsulates(h, g):
                        observed_nom += 2 * get_length(g)
                        observed_denom += 1
                        weight += get_length(g)

                    elif g[2] == empty_tag and h[2] != empty_tag and encapsulates(g, h):
                        observed_nom += 2 * get_length(h)
                        observed_denom += 1
                        weight += get_length(h)

                    elif g[2] == empty_tag and h[2] == empty_tag:
                        empty_count += intersect

    if observed_nom == 0:
        return 1.0, weight

    observed = float(observed_nom / observed_denom)
    entities = [i for x in in_entities
                for i in x if i[2] != empty_tag]
    expected = 0.0
    for g in entities:
        seenItself = False
        for h in entities:
            # TODO: See if this comparison is correct? Since two annotators can agree on some annotation,
            # g and h might come from different annotators but still be exactly same.
            if g == h and not seenItself:
                seenItself = True
                continue

            leng = get_length(g)
            lenh = get_length(h)
            expected_nom += leng * leng + lenh * lenh + leng * lenh * get_metric(g, h)
            expected_denom += leng + lenh

    if expected_denom != 0:
        expected = float(expected_nom / expected_denom)

    if expected == 0:
        return 0.0, weight

    return 1.0 - observed / expected, weight


def get_result(all_df, doc_length, overlap=False, empty_tag="empty"):
    annotators = all_df.annotator.unique().tolist()
    foliasets = all_df.focus.unique().tolist()
    kAlpha_sets = {}

    if not overlap:
        for foliaset in foliasets:
            all_entities = []
            for annot in annotators:
                entities = all_df[(all_df.annotator == annot) & (
                    all_df.focus == foliaset)].entity.tolist()
                entities = sorted(entities)
                entities = organize_data(entities, doc_length, empty_tag)
                all_entities.append(entities)

            kAlpha_sets[foliaset] = calculate_kalpha(all_entities, annotators, empty_tag)

    else:
        for foliaset in foliasets:
            curr_df = all_df[all_df.focus == foliaset]
            tags = curr_df.tag.unique()
            kAlpha_tags = {}
            for tag in tags:
                all_entities = []
                for annot in annotators:
                    entities = curr_df[(curr_df.annotator == annot) & (
                        curr_df.tag == tag)].entity.tolist()
                    entities = sorted(entities)
                    entities = organize_data(entities, doc_length, empty_tag)
                    all_entities.append(entities)

                kAlpha_tags[tag] = calculate_kalpha(all_entities, annotators, empty_tag)

            kAlpha_sets[foliaset] = kAlpha_tags

    return kAlpha_sets

def resolve_entity_discontinuity(entity_ids):
    """
    entity_ids is a sorted list of sentence and word ids of the entity.
    If tokens in the entity span are discontinuous, divides into multiple continuous parts.
    Returns a list of entities, each containing entity ids.
    """

    if len(entity_ids) == 1:
        return [entity_ids]

    entities = []
    idx = 0
    up_to_idx = 0
    while idx < len(entity_ids) - 1:
        curr_word_id = entity_ids[idx][1]
        next_word_id = entity_ids[idx+1][1]

        if curr_word_id + 1 != next_word_id: # if there is a discontinuity
            entities.append(entity_ids[up_to_idx:idx+1])
            up_to_idx = idx+1

        idx += 1

    entities.append(entity_ids[up_to_idx:])

    return entities

def convert_folia_to_common_format(folia_doc, df, sentence_starting_points, annot,
                                   discard_tokens_in_negative_sentences=False):
    trigger_list = ["etype", "emention"]

    sent_num = 0
    start_point = 0
    for paragraph in folia_doc.paragraphs():
        for sentence in paragraph.sentences():
            curr_sent_entities = []
            for layer in sentence.select(folia.EntitiesLayer):
                for entity in layer.select(folia.Entity):
                    entity_ids = []
                    for wref in entity.wrefs():
                        match = re.search(r'(p\.\d+\.s\.\d+)\.w\.(\d+)$', wref.id)
                        entity_ids.append([match.group(1), # paragraph and sentence
                                           int(match.group(2)) - 1]) # word, -1 because ids are 1-indexed

                    entity_ids = sorted_nicely(entity_ids)
                    curr_sent_entities.append([entity_ids, entity.cls, entity.set])

            # NOTE: This part is unique to our project, so ignore this "if".
            # Check if the current sentence is positive (has trigger)
            if discard_tokens_in_negative_sentences and any([a[1] in trigger_list for a in curr_sent_entities]):
                continue

            for entity in curr_sent_entities:
                # Check if entity spans accross sentences, if so divide into separate entities
                entity_ids = entity[0]
                unique_sents = list(set([idx[0] for idx in entity_ids]))
                if len(unique_sents) > 1: # entity spans accross multiple sentences
                    curr_entities = []
                    for sent in unique_sents:
                        curr_entity_ids = [idx for idx in entity_ids if idx[0] == sent]
                        curr_entities.extend(resolve_entity_discontinuity(curr_entity_ids))
                else: # entity resides in a single sentence
                    curr_entities = resolve_entity_discontinuity(entity_ids)

                for curr_sent_ids in curr_entities:
                    curr_sent_starting_point = sentence_starting_points[curr_sent_ids[0][0]]
                    curr_starting_id = curr_sent_starting_point + curr_sent_ids[0][1]
                    curr_ending_id = curr_sent_starting_point + curr_sent_ids[-1][1]
                    df = df.append({"entity": [curr_starting_id, curr_ending_id, entity[1]],
                                    "annotator": annot, "focus": entity[2], "tag": entity[1]},
                                   ignore_index=True)

            sent_num += 1

    return df

def get_ssp_and_doc_length(doc):
    sentence_starting_points = {}
    doc_length = 0
    for paragraph in doc.paragraphs():
        for sentence in paragraph.sentences():
            curr_para_sent_id = re.search(r'(p\.\d+\.s\.\d+)$', sentence.id).group(1)
            sentence_starting_points[curr_para_sent_id] = doc_length
            doc_length += len(sentence)

    return sentence_starting_points, doc_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a script for calculating Krippendorff's Alpha between two or more annotators. You can find the latest paper on the metric in this link : http://web.asc.upenn.edu/usr/krippendorff/m-Replacement%20of%20section%2012.4%20on%20unitizing%20continua%20in%20CA,%203rd%20ed.pdf (Please give document(s) first, then pass options.)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--empty', type=str, help="If you have a category named empty pass this argument with a string.",
                        default="empty", action='store', required=False)
    parser.add_argument('-F', '--Folia', help="If your inputs are folia documents",
                        action='store_true', default="", required=False)
    # This overlap setting assumes that one tag doesn't apply to same word more than once
    parser.add_argument('-o', '--overlap', help="If your categories(tags) are overlapping",
                        action='store_true', default="", required=False)
    parser.add_argument('-d', '--discard_neg_sents', help="Discard tokens in negative sentences",
                        action='store_true', default="", required=False)
    # For the csv file's format see https://github.com/emerging-welfare/kAlpha/example.csv
    parser.add_argument('document', nargs='+',
                        help="CSV File For the csv file's format see https://github.com/OsmanMutlu/kAlpha/example.csv (You can choose multiple Folia docs too.)")
    args = parser.parse_args()

    if args.Folia:

        all_df = pd.DataFrame()
        for i, docfile in enumerate(args.document):
            doc = folia.Document(file=docfile)
            # if first annotator get sentence starting indexes and document length.
            # All documents' sentences should be same length.
            if i == 0:
                sentence_starting_points, doc_length = get_ssp_and_doc_length(doc)

            all_df = convert_folia_to_common_format(doc, all_df, sentence_starting_points, "annot" + str(i + 1),
                                                    discard_tokens_in_negative_sentences=args.discard_neg_sents)

    else:
        if len(args.document) != 1:
            sys.exit("Please give only one document for CSV input.")

        all_df = pd.read_csv(args.document[0], header=None, names=[
                             'annotator', 'start', 'end', 'tag'], comment="#")
        all_df["focus"] = "result" # This is just a placeholder
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
            lambda row: (row.start, row.end, row.tag), axis=1)

    # We are finished with getting data. Both folia's and csv's data layout is same
    result = get_result(all_df, doc_length, overlap=args.overlap, empty_tag=args.empty)
    print("Krippendorff's Alpha is : " + str(result))
