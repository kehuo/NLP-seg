import codecs
import gensim


def get_sentences(rfile):
    sentences = []
    f = codecs.open(rfile, encoding='utf-8', mode='r')
    for line in f:
        line.rstrip()
        sentences.append(line)
    f.close()
    return sentences


def main():
    sentences = get_sentences("/Users/hk/dev/NLP-seg/nlp_seg/data/corpus/total_corpus.txt")
    print("read %s sentences." % len(sentences))
    model = gensim.models.Word2Vec(sentences, size=120, window=5, min_count=0, workers=4)
    model.wv.save_word2vec_format("vectors.txt")


if __name__ == "__main__":
    main()
