'''
Copyright (c) 2018 MD ATAUR RAHMAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


# Read the Excel File and copy the label and document
def generateTextFile(filename):

    document = []
    label = []

    df = pd.read_excel(filename, sheet_name='Sheet1')

    print("Column headings:")
    print(df.columns)

    # print(df['Unnamed: 2'])

    for lbl, doc in zip(df['Unnamed: 2'][8:], df['Unnamed: 1'][8:]):
        lbl = str(lbl)
        doc = str(doc)

        if check_Labels(lbl):
            label.append(lbl)
            document.append(doc)
            # print(str(lbl)+" "+str(doc))

    return document, label


# Append the contents of 3 files in a single document and labels
def append_Doc_Lbl(document, label, doc_new, lbl_new):

    for doc, lbl in zip(doc_new, lbl_new):
        document.append(doc)
        label.append(lbl)

    return document, label


# Checking and correcting the labels
def check_Labels(lbl):

    # There are also some rubbish labels so a simple string compare with empty field won't work
    if lbl == 'sad' or lbl == 'happy' or lbl == 'disgust' or lbl == 'surprise' or lbl == 'fear' or lbl == 'angry':
        return True

    return False

def create_balanced_dataset(label, document):

    out_train = []
    out_test = []
    everything_else = []
    count = [0, 0, 0, 0, 0 ,0]
    # First 300 to trainset, upto second 100 to testset, everything else to separate list
    for lbl, doc in zip(label, document):
        # out.append(lbl+" "+doc)
        if lbl == 'sad':
            if count[0]<1000:
                out_train.append(lbl+" "+doc)
            elif count[0]<1000+200:
                out_test.append(lbl+" "+doc)
            else:
                everything_else.append(lbl+" "+doc)
            count[0]+=1

        elif lbl == 'happy':
            if count[1]<1500:
                out_train.append(lbl+" "+doc)
            elif count[1]<1500+300:
                out_test.append(lbl+" "+doc)
            else:
                everything_else.append(lbl+" "+doc)
            count[1]+=1

        elif lbl == 'disgust':
            if count[2]<500:
                out_train.append(lbl+" "+doc)
            elif count[2]<500+100:
                out_test.append(lbl+" "+doc)
            else:
                everything_else.append(lbl+" "+doc)
            count[2]+=1

        elif lbl == 'surprise':
            if count[3]<400:
                out_train.append(lbl+" "+doc)
            elif count[3]<400+80:
                out_test.append(lbl+" "+doc)
            else:
                everything_else.append(lbl+" "+doc)
            count[3]+=1

        elif lbl == 'fear':
            if count[4]<300:
                out_train.append(lbl+" "+doc)
            elif count[4]<300+60:
                out_test.append(lbl+" "+doc)
            else:
                everything_else.append(lbl+" "+doc)
            count[4]+=1

        elif lbl == 'angry':
            if count[5]<1000:
                out_train.append(lbl+" "+doc)
            elif count[5]<1000+200:
                out_test.append(lbl+" "+doc)
            else:
                everything_else.append(lbl+" "+doc)
            count[5]+=1

    return out_train, out_test, everything_else

def main():

    document = []
    label = []

    doc, lbl = generateTextFile('corpus/EkattorTV_Comments.xlsx')
    document, label = append_Doc_Lbl(document, label, doc, lbl)

    doc, lbl = generateTextFile('corpus/Magistrate_Comments.xlsx')
    document, label = append_Doc_Lbl(document, label, doc, lbl)

    doc, lbl = generateTextFile('corpus/ImranHSarker_Comment.xlsx')
    document, label = append_Doc_Lbl(document, label, doc, lbl)

    # for doc, lbl in zip(doc[:10], lbl[:10]):
    #     print(str(lbl)+" "+str(doc))

    print("Total Number of Documents and Labels:")
    print("Documents = {0}".format(len(document)))
    print("Labels = {0}".format(len(label)))


    # combining the label and documents in the same string
    # out = []
    # for lbl, doc in zip(label, document):
    #     out.append(lbl+" "+doc)
    #
    # # creating the corpus_all.txt file
    # with open("corpus_all.txt", 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(out))

    # Creating a balanced training set
    out_train, out_test, everything_else = create_balanced_dataset(label, document)

    # creating the separate training, testing and other file
    with open("train.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_train))

    with open("test.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_test))

    with open("everything_else.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(everything_else))


if __name__ == '__main__':
    main()


