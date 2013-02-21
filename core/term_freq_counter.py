import os
import re
import nltk

def main():
    fin = os.popen("awk -F',' '{print $3}' ~/Workspace/grass_mud_horse/train/Train_small.csv").readlines()
    fin = ' '.join(map(lambda x: x.strip(), fin))
    tokens = nltk.word_tokenize(fin)
    text = nltk.Text(tokens)
    freq = nltk.FreqDist(text)
    print text.collocations(20)
    freq.plot(30)
if __name__ == "__main__":
    main()
